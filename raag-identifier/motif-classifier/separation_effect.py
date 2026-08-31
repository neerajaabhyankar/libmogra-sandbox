"""Does source separation help classification, or only the pitch track?

`../source-separation/inspect_separation.py` shows HPSS makes CREPE's output cleaner —
more voiced frames, ~10x less jitter, a peakier histogram. That is a claim about the
*tracker*, not about *accuracy*. This runs the strong methods over both caches, changing
nothing else, so the two questions stay separate.

    poetry run python extract.py --tracker crepe --separate hpss   # build the cache first
    poetry run python separation_effect.py
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.extract import DATA_VERSION, cache_path
from represent import Params
from tune import cv_score

RESULTS = Path(__file__).resolve().parent / "results" / DATA_VERSION

BASE = dict(tracker="tony", note_source="hmm", tonic_mode="true", tonic_refine=True,
            min_dur=0.0, max_cents_dev=2400.0, collapse_repeats=True)
HIST = dict(n_bins=120, source="frames", tracker="crepe", metric="chi2",
            smooth=1.0, power=0.5, tonic_mode="true")
SURF = dict(tracker="crepe", n_bins=80, tau=0.3, smooth=1.0, tonic_mode="true", metric="chi2")
M13 = dict(lam_db=0.3, which="bigram_dur", w_uni=0.3)

# Only the crepe-side methods can move: `separate` selects which crepe cache the melody
# surface / histogram reads. M13 runs off Tony's notes, which are not re-extracted here, so
# it is the control — its row should not move at all, and if it does something is wired wrong.
METHODS = {
    "m9  melody surface":  ("m9",  lambda sep: dict(SURF, separate=sep), ()),
    "m11 histogram":       ("m11", lambda sep: dict(HIST, separate=sep), ()),
    "m12 histogram+DB":    ("m12", lambda sep: dict(HIST, separate=sep, lam=0.3), ()),
    "m13 bigram LM (ctrl)": ("m13", lambda sep: dict(M13), ()),
    "m14 M12+M13":         ("m9plus", lambda sep: dict(
        w_tdms=3.0, base="m13", base_kw=dict(M13), tdms_cls="m12",
        tdms_kw=dict(HIST, separate=sep, lam=0.3)), ()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=[None, "hpss"])
    args = ap.parse_args()

    backends = [None if b in ("none", "None") else b for b in args.backends]
    for b in backends:
        p = cache_path("crepe", b)
        if not p.exists():
            raise SystemExit(f"missing cache {p}\n  build it: poetry run python extract.py "
                             f"--tracker crepe --separate {b}")

    rows = []
    hdr = f"{'method':<22} " + " ".join(f"{str(b or 'none'):>10}" for b in backends) + f"{'delta':>9}"
    print(hdr); print("-" * len(hdr))
    for label, (name, kwf, extra) in METHODS.items():
        got = {}
        for b in backends:
            # The representation (Tony's notes) is deliberately NOT switched: only the
            # crepe-side methods take `separate`, so M13 — which reads Tony — is a true
            # control whose row must not move. Switching the rep too would change both
            # halves at once and the comparison would measure nothing in particular.
            rep = Params(**BASE)
            got[b] = cv_score(name, rep, kwf(b), extra_trackers=extra)
        base = got[backends[0]]["top1"]
        line = f"{label:<22} " + " ".join(f"{got[b]['top1']:10.3f}" for b in backends)
        print(line + f"{got[backends[-1]]['top1'] - base:+9.3f}", flush=True)
        rows.append({"method": label, **{str(b): got[b] for b in backends}})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "separation_effect.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {RESULTS / 'separation_effect.json'}")


if __name__ == "__main__":
    main()
