"""What does the hand-annotated tonic actually buy?

v1 changed two things at once — clips went from ~6 s to 20-60 s, and every recording gained
a hand-annotated Sa. This runs the strong methods across tonic policies on identical audio,
so the tonic effect is isolated from the length effect (the length effect is isolated
separately by re-running the same method under RAAG_DATA_VERSION=v0).

    poetry run python tonic_ablation.py --methods m3 m4 m9 m9plus m7
"""

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.extract import DATA_VERSION
from represent import Params
from tune import cv_score

HERE = Path(__file__).resolve().parent
# v0 results stay exactly where plan.md links them; v1 gets its own subdirectory so the
# two dataset versions never overwrite each other's sweeps.
RESULTS = HERE / "results" / ("" if DATA_VERSION == "v0" else DATA_VERSION)
RESULTS.mkdir(parents=True, exist_ok=True)

BASE_REP = {"tracker": "tony", "note_source": "hmm", "tonic_refine": True,
            "min_dur": 0.0, "max_cents_dev": 2400.0, "collapse_repeats": True}

# tonic policies, cheapest-to-best guess order. "true" is the v1 annotation.
TONICS = ["video", "chroma_video", "true"]

# each method's tuned config from the v0 sweeps — held fixed so the only thing moving
# between rows is the tonic (and, versus v0, the audio)
METHODS = {
    "m3": (dict(shift_mode="none", w_arohana=1.0, symmetric=False, lam_bi=0.7, lam_uni=0.25,
                uni_from_scale=0.75, nyas_boost=1.0, w_dur=0.5, dur_weighted=False, w_skip=0.5), ()),
    "m4": (dict(w_crepe=1.0, primary="tony"), ("crepe",)),
    "m9": (dict(tracker="crepe", n_bins=60, tau=0.15, smooth=1.0), ()),  # tonic_mode injected below
    "m9plus": (dict(w_tdms=0.5, base="m4", base_kw=dict(w_crepe=1.0, primary="tony"),
                    tdms_kw=dict(tracker="crepe", n_bins=60, tau=0.3)), ("crepe",)),
    "m10": (dict(w_emph=1.0, w_reg=0.5, w_vivadi=1.0, vaadi_w=3.0, samvaadi_w=2.0), ()),
    "m7": (dict(use_channel=True, channel_kw=dict(p_self=0.2, prior=5.0, emission_temp=0.3, w_dur=1.0),
                base_kw=dict(lam_bi=0.7, lam_uni=0.1, uni_from_scale=0.75),
                w_crepe=1.0, calibrate="zscore"), ("crepe",)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["m3", "m4", "m9", "m9plus"])
    ap.add_argument("--tonics", nargs="+", default=TONICS)
    ap.add_argument("--out", default="tonic_ablation.json")
    args = ap.parse_args()

    rows = []
    for name in args.methods:
        method_kw, extra = METHODS[name]
        for tonic in args.tonics:
            rep = Params(**BASE_REP, tonic_mode=tonic)
            # M9/M9+ build their surface from a second tracker's frames and look the tonic
            # up themselves, so the policy has to be handed to them explicitly
            method_kw = dict(method_kw)
            if name == "m9":
                method_kw["tonic_mode"] = tonic
            elif name == "m9plus":
                method_kw["tdms_kw"] = dict(method_kw["tdms_kw"], tonic_mode=tonic)
            t0 = time.time()
            try:
                sc = cv_score(name, rep, method_kw, extra_trackers=extra)
            except Exception as e:
                print(f"  FAILED {name}/{tonic}: {type(e).__name__}: {e}", flush=True)
                continue
            rows.append({"method": name, "tonic": tonic, **sc})
            print(f"{name:8s} {tonic:14s} top1 {sc['top1']:.3f}±{sc['top1_std']:.3f}  "
                  f"top5 {sc['top5']:.3f}  mrr {sc['mrr']:.3f}  vid {sc['video_top1']:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            (RESULTS / args.out).write_text(json.dumps(rows, indent=1))
    print(f"\n-> {RESULTS / args.out}")


if __name__ == "__main__":
    main()
