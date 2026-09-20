"""Accuracy on the held-out test split, against the numbers this model published.

    python  tests/test_model.py                # all 150 test clips, the full report
    python  tests/test_model.py --limit 20     # a quick look
    pytest  tests/test_model.py                # a 20-clip sanity check

`weights/test_metrics.json` records what was measured when the model was trained — top-1
0.48 and top-5 0.82 over 150 clips. This re-measures it from audio, so a refactor that
quietly moves the model has somewhere to show up.

**It is slow.** CREPE at a 10 ms hop dominates: budget roughly fifteen seconds a clip, so
the full 150 is well over half an hour. `--limit` exists for that reason, and the pytest
entry point uses it.

Audio comes from a local corpus if the surrounding repository is there, and otherwise from
the pinned Hugging Face revision `raag_fusion.data` names — so this works for someone who
downloaded the model and has nothing else.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from raag_fusion import RaagIdentifier                    # noqa: E402
from raag_fusion import audio as A                        # noqa: E402

WEIGHTS = ROOT / "weights"
PUBLISHED = WEIGHTS / "test_metrics.json"

# Where the corpus sits when the whole repository is present. Absent for a Hub download,
# which falls back to `raag_fusion.data.stream`.
LOCAL_AUDIO = ROOT.parent / "hindustani-raag-small-v1"
LOCAL_TONICS = ROOT.parent / "raagdataset/tonics.csv"

CLIP_RE = re.compile(r"^(train|test)_\[(.+)\]_chunk(\d+)\.mp3$")
PYTEST_LIMIT = 20            # keeps `pytest` to a couple of minutes rather than an hour
TOLERANCE = 0.02             # how far the full run may drift from the published numbers


def local_clips():
    """(raag, path, tonic_hz) for the test split, from the corpus next to this repository."""
    if not (LOCAL_AUDIO.is_dir() and LOCAL_TONICS.exists()):
        return None
    tonics = {}
    for line in LOCAL_TONICS.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) > 2 and parts[0]:
            tonics[parts[0]] = float(parts[2])

    out = []
    for path in sorted(LOCAL_AUDIO.glob("*/test_*.mp3")):
        m = CLIP_RE.match(path.name)
        if m and m.group(2) in tonics:
            out.append((path.parent.name, path, tonics[m.group(2)]))
    return out or None


def hub_clips():
    """The same, streamed from the pinned dataset revision."""
    import io

    from raag_fusion import data

    out = []
    for raag, name, mp3, tonic in data.stream():
        m = CLIP_RE.match(name)
        if m and m.group(1) == "test":
            out.append((raag, io.BytesIO(mp3), float(tonic)))
    return out or None


def clips(limit=None):
    found = local_clips() or hub_clips()
    if not found:
        raise SystemExit(
            "no test audio. Either keep this model directory inside the raag-identifier "
            "repository, or `pip install datasets` so it can be streamed from the Hub.")
    found.sort(key=lambda c: str(c[1]))
    if limit:
        # spread the sample across raags rather than taking the alphabetical head
        step = max(1, len(found) // limit)
        found = found[::step][:limit]
    return found


def evaluate(model, items, progress=None):
    """(top1, top5, n) over `items`."""
    hits1 = hits5 = 0
    for i, (raag, source, tonic) in enumerate(items, 1):
        y, sr = A.load(source)
        names = [p.raag for p in model.predict(y, sr, tonic, top_k=5)]
        hits1 += names[0] == raag
        hits5 += raag in names
        if progress:
            progress(i, len(items), raag, names)
    n = len(items)
    return hits1 / n, hits5 / n, n


def published():
    return json.loads(PUBLISHED.read_text()) if PUBLISHED.exists() else None


# ------------------------------------------------------------------------ pytest

def test_accuracy_is_clearly_better_than_chance():
    """A sample, not the full set. Fifty raags means chance is 0.02; anything that still
    ranks the right raag first on a fifth of a sample is not a broken model."""
    model = RaagIdentifier.load(device="cpu")
    top1, top5, n = evaluate(model, clips(limit=PYTEST_LIMIT))
    assert n == PYTEST_LIMIT
    assert top1 >= 0.15, f"top-1 {top1:.3f} over {n} clips"
    assert top5 >= 0.45, f"top-5 {top5:.3f} over {n} clips"


def test_published_metrics_are_present_and_sane():
    p = published()
    assert p is not None, "weights/test_metrics.json is missing"
    assert p["n_clips"] == 150
    assert 0.0 < p["top1"] <= p["top5"] <= 1.0


# ------------------------------------------------------------------------ script

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="use a sample this size")
    ap.add_argument("--quiet", action="store_true", help="just the totals")
    a = ap.parse_args()

    items = clips(limit=a.limit)
    source = "local corpus" if local_clips() else "the Hub"
    print(f"{len(items)} test clips from {source}\n")

    model = RaagIdentifier.load(device="cpu")

    def show(i, total, raag, names):
        if not a.quiet:
            mark = "1" if names[0] == raag else ("5" if raag in names else ".")
            print(f"  [{i:>3}/{total}] {mark}  {raag:<22} -> {names[0]}")

    top1, top5, n = evaluate(model, items, progress=show)

    print(f"\n  top-1  {top1:.4f}   ({round(top1 * n)}/{n})")
    print(f"  top-5  {top5:.4f}   ({round(top5 * n)}/{n})")

    p = published()
    if p and n == p["n_clips"]:
        d1, d5 = top1 - p["top1"], top5 - p["top5"]
        print(f"\n  published: top-1 {p['top1']:.4f}, top-5 {p['top5']:.4f}")
        print(f"  drift:     {d1:+.4f}          {d5:+.4f}")
        if max(abs(d1), abs(d5)) > TOLERANCE:
            print(f"\n  OVER the {TOLERANCE} tolerance -- something moved the model")
            return 1
        print("\n  matches what this model published")
    elif p:
        print(f"\n  (published numbers are over {p['n_clips']} clips; this was {n}, "
              f"so they are not comparable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
