"""How good was the tonic heuristic, now that we have ground truth?

Every method before v1 estimated Sa from the audio, and `ceilings.py` said an oracle tonic
roughly tripled top-1. v1 ships a hand annotation, so the estimate can finally be scored
directly instead of inferred from downstream accuracy.

    poetry run python tonic_audit.py --tracker tony
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.extract import DATA_VERSION
from utils.extract import list_clips, load_cache
from represent import Params, _load

HERE = Path(__file__).resolve().parent
# v0 results stay exactly where plan.md links them; v1 gets its own subdirectory so the
# two dataset versions never overwrite each other's sweeps.
RESULTS = HERE / "results" / ("" if DATA_VERSION == "v0" else DATA_VERSION)
RESULTS.mkdir(parents=True, exist_ok=True)

MODES = [
    ("clip", True, "per-clip, refined"),
    ("video", True, "per-video, refined"),
    ("chroma_clip", True, "per-clip + Sa-vs-Pa correction"),
    ("chroma_video", True, "per-video + Sa-vs-Pa correction"),
]


def audit(tracker):
    clips = {c["clip_id"]: c for c in list_clips()}
    truth_video = {c["video"]: c["true_tonic_hz"] for c in clips.values()}
    rows = []
    for mode, refine, label in MODES:
        _, meta, clip_t, video_t = _load(
            tracker, mode, refine, 2400.0,
            (("alpha", 0.6), ("beta", 0.9), ("gamma", 0.05), ("median_target", 6.0)),
        )
        per_clip = mode in ("clip", "chroma_clip")
        errs, semis = [], []
        for c in meta:
            est = clip_t[c["clip_id"]] if per_clip else video_t[c["video"]]
            true = truth_video[c["video"]]
            cents = 1200.0 * np.log2(est / true)
            errs.append(cents)
            # nearest semitone of error, and the leftover after removing it
            semis.append(int(np.round(cents / 100.0)))
        errs = np.asarray(errs)
        semis = np.asarray(semis)
        resid = errs - 100.0 * semis
        # Every method downstream is octave-folded, so an octave error costs nothing and
        # the number that predicts accuracy is the error mod 12. Raw error is reported
        # alongside it because it is what "is the annotation right?" actually means.
        pc = np.mod(semis, 12)
        pc_signed = np.where(pc > 6, pc - 12, pc)
        rows.append({
            "mode": mode, "label": label, "n": len(errs),
            # --- pitch-class (what the octave-folded methods actually see)
            "pc_exact": float(np.mean(pc == 0)),
            "pc_within_1_semitone": float(np.mean(np.abs(pc_signed) <= 1)),
            "pc_off_by_fifth": float(np.mean(pc == 7)),
            "pc_off_by_fourth": float(np.mean(pc == 5)),
            "pc_hist": {str(k): int(v) for k, v in
                        sorted(Counter(pc_signed).items(), key=lambda kv: -kv[1])[:6]},
            # --- absolute (is the annotated Hz recovered at all?)
            "within_50c": float(np.mean(np.abs(errs) < 50)),
            "median_abs_cents": float(np.median(np.abs(errs))),
            "off_by_octave": float(np.mean(np.isin(semis, (12, -12, 24, -24)))),
            "semitone_hist": {str(k): int(v) for k, v in
                              sorted(Counter(semis).items(), key=lambda kv: -kv[1])[:8]},
            # once the pitch class is right, how well is the fine tuning recovered?
            "resid_mad_cents": float(np.median(np.abs(resid[pc == 0]))) if (pc == 0).any() else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="tony")
    args = ap.parse_args()
    rows = audit(args.tracker)
    hdr = (f"{'tonic policy':34s} {'pc exact':>9s} {'pc +-1':>7s} {'pc 5th':>7s} "
           f"{'|abs|<50c':>10s} {'8ve err':>8s} {'fine MAD':>9s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:34s} {r['pc_exact']:9.3f} {r['pc_within_1_semitone']:7.3f} "
              f"{r['pc_off_by_fifth']:7.3f} {r['within_50c']:10.3f} {r['off_by_octave']:8.3f} "
              f"{(r['resid_mad_cents'] or float('nan')):8.0f}c")
    print("\n  pc exact   = tonic lands on the right pitch class (what folded methods see)")
    print("  |abs|<50c  = the annotated Hz is recovered, octave included")
    print("  fine MAD   = median |error| in cents among the pitch-class-correct ones\n")
    for r in rows:
        print(f"{r['mode']:14s} pitch-class error: {r['pc_hist']}")
    out = RESULTS / f"tonic_audit_{args.tracker}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
