"""Hyperparameter search on train only, scored by grouped (by-video) cross-validation.

Nothing here is fitted in the machine-learning sense — every method is prescriptive, built
from the libmogra database. What gets chosen are a handful of scalars: which note source,
which tonic policy, how much the scale term counts against the phrase term, how the
grammar is smoothed. The grouped CV is what stops those scalars from being chosen to fit
one performer's recording.

    poetry run python tune.py --stage rep      # representation knobs, method held fixed
    poetry run python tune.py --stage m2
    poetry run python tune.py --stage m3
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.extract import DATA_VERSION
from evaluate import evaluate, evaluate_by_video, group_folds, make_method
from features import build_features
from dataclasses import replace

from features import align_trackers, estimate_tuning_offsets
from represent import Params, build_clips

HERE = Path(__file__).resolve().parent
# v0 results stay exactly where plan.md links them; v1 gets its own subdirectory so the
# two dataset versions never overwrite each other's sweeps.
RESULTS = HERE / "results" / ("" if DATA_VERSION == "v0" else DATA_VERSION)
RESULTS.mkdir(parents=True, exist_ok=True)

_FEAT_CACHE = {}

# Chosen by `--stage rep` (see plan.md): Tony's own note HMM, tonic pooled per video with
# the sub-semitone refinement, no duration filter, notes kept out to +/-2 octaves.
DEFAULT_REP = {
    "tracker": "tony",
    "note_source": "hmm",
    "tonic_mode": "video",
    "tonic_refine": True,
    "min_dur": 0.0,
    "max_cents_dev": 2400.0,
}


def split_feats(rep: Params, feature_kw, split="train", extra_trackers=()):
    """Features for one split. With `extra_trackers`, each clip comes back as a
    MultiFeatures carrying every tracker's view of it (M4/M7 need both)."""
    key = (rep, tuple(sorted(feature_kw.items())), split, tuple(extra_trackers))
    if key not in _FEAT_CACHE:
        kw = dict(feature_kw)
        # `tuning_offsets: True` means "estimate them", which has to happen from TRAIN clips
        # regardless of which split we are building — and it must be resolved here rather
        # than passed in, so the cache key stays hashable.
        if kw.pop("tuning_offsets", False):
            from utils.raagdb import dataset_raags

            base = build_clips(rep)
            kw["tuning_offsets"] = estimate_tuning_offsets(
                [c for c in base if c.split == "train"],
                scales={f: r.scale for f, r in dataset_raags().items()},
            )

        def one(tracker):
            r = replace(rep, tracker=tracker)
            return build_features([c for c in build_clips(r) if c.split == split], **kw)

        if not extra_trackers:
            _FEAT_CACHE[key] = one(rep.tracker)
        else:
            sets = {t: one(t) for t in (rep.tracker, *extra_trackers)}
            _FEAT_CACHE[key] = align_trackers(sets, rep.tracker)
    return _FEAT_CACHE[key]


def train_feats(rep: Params, feature_kw, extra_trackers=()):
    return split_feats(rep, feature_kw, "train", extra_trackers)


def cv_score(method_name, rep: Params, method_kw, feature_kw=None, n_folds=5, seed=0,
             extra_trackers=()):
    """Grouped CV. The methods have no fitted state, so folds differ only in which clips
    are scored — which is exactly what we want to average over, and it also gives an
    honest spread rather than one lucky number."""
    feature_kw = feature_kw or {}
    feats = train_feats(rep, feature_kw, extra_trackers)
    fold_of_video = group_folds([f.clip for f in feats], n_folds=n_folds, seed=seed)
    method = make_method(method_name, **method_kw)
    per_fold = []
    all_rows = []
    for k in range(n_folds):
        sub = [f for f in feats if fold_of_video[f.clip.video] == k]
        if not sub:
            continue
        if method.fitted:
            # anything estimated from labels is refit on the other folds only, so the
            # held-out clips never touch the model that scores them
            method = make_method(method_name, **method_kw)
            method.fit([f for f in feats if fold_of_video[f.clip.video] != k])
        m, rows = evaluate(method, sub)
        per_fold.append(m)
        all_rows += rows
    out = {k: float(np.mean([m[k] for m in per_fold])) for k in ("top1", "top5", "mrr")}
    out["top1_std"] = float(np.std([m["top1"] for m in per_fold]))
    out["video_top1"] = evaluate_by_video(all_rows, method.raags)[0]
    return out


def grid(**kw):
    keys = list(kw)
    for vals in itertools.product(*(kw[k] for k in keys)):
        yield dict(zip(keys, vals))


def sweep(name, method_name, rep_grid, method_grid, feature_grid=None, top=12,
          extra_trackers=()):
    rows = []
    feature_grid = feature_grid or [{}]
    combos = [
        (r, m, f)
        for r in rep_grid
        for m in method_grid
        for f in feature_grid
    ]
    print(f"[{name}] {len(combos)} configs")
    t0 = time.time()
    for i, (rk, mk, fk) in enumerate(combos):
        rep = Params(**rk)
        try:
            sc = cv_score(method_name, rep, mk, feature_kw=fk, extra_trackers=extra_trackers)
        except Exception as e:
            print(f"  FAILED {rk} {mk}: {type(e).__name__}: {e}")
            continue
        rows.append({"rep": rk, "method": mk, "features": fk,
                     "extra_trackers": list(extra_trackers), **sc})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(combos)}  {(time.time()-t0)/(i+1):.1f}s/config", flush=True)
    rows.sort(key=lambda r: -r["top1"])
    out = RESULTS / f"sweep_{name}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"[{name}] best {top}:")
    for r in rows[:top]:
        extra = {**r["rep"], **r["method"], **r["features"]}
        print(f"  top1 {r['top1']:.3f}±{r['top1_std']:.3f}  top5 {r['top5']:.3f}  "
              f"vid {r['video_top1']:.3f}  {extra}")
    print(f"  -> {out}")
    return rows


# ---------------------------------------------------------------- stages


def stage_rep():
    """Which transcription/tonic representation to build everything on. Method held at a
    mid-range M3 so the comparison is about the representation, not the matcher."""
    rep_grid = list(
        grid(
            tracker=["tony"],
            note_source=["hmm", "segment"],
            tonic_mode=["clip", "video", "chroma_video"],
            tonic_refine=[False, True],
            min_dur=[0.0, 0.15, 0.3],
            collapse_repeats=[True, False],
            max_cents_dev=[1800.0, 2400.0],
        )
    )
    return sweep("rep", "m3", rep_grid, [{"shift_mode": "none"}])


def _rep_variants(rep_kw):
    """The chosen representation, with `collapse_repeats` left open: it interacts with the
    method (an exact matcher needs the collapsed form; a grammar may prefer the repeats)."""
    return [dict(rep_kw, collapse_repeats=c) for c in (True, False)]


def stage_m1(rep_kw):
    return sweep(
        "m1",
        "m1",
        _rep_variants(rep_kw),
        list(grid(shift_mode=["none", "global", "per_raag"], length_bonus=[0.0, 0.02, 0.1])),
    )


def stage_m2(rep_kw):
    return sweep(
        "m2",
        "m2",
        _rep_variants(rep_kw),
        list(
            grid(
                shift_mode=["none"],
                n_max=[3, 4],
                idf_power=[0.0, 1.0, 2.0],
                w_arohana=[0.0, 0.3, 1.0],
                w_scale=[0.0, 0.5, 2.0, 5.0],
                len_power=[0.0, 1.0],
                norm=["raag_l1", "raag_l2"],
            )
        ),
        feature_grid=list(grid(max_skip=[0, 1, 2])),
    )


def stage_m3(rep_kw):
    return sweep(
        "m3",
        "m3",
        _rep_variants(rep_kw),
        list(
            grid(
                shift_mode=["none"],
                w_arohana=[0.25, 0.5, 1.0],
                symmetric=[False, True],
                lam_bi=[0.5, 0.7, 0.9],
                lam_uni=[0.1, 0.25],
                uni_from_scale=[0.25, 0.5, 0.75],
                nyas_boost=[0.0, 1.0],
                w_dur=[0.0, 0.5, 1.0],
                dur_weighted=[False, True],
                w_skip=[0.0, 0.5],
            )
        ),
    )


def stage_m4(rep_kw):
    """Lever (a): does CREPE see phrase evidence Tony's note HMM pruned away?"""
    return sweep(
        "m4", "m4", [dict(rep_kw, collapse_repeats=True)],
        list(grid(w_crepe=[0.0, 0.3, 0.6, 1.0, 1.5, 2.0], primary=["tony", "crepe"]))
        + [{"w_crepe": 0.0, "primary": "crepe"}],
        extra_trackers=("crepe",),
    )


def stage_m5(rep_kw):
    """Lever (b): kan-swars / meends as a noisy channel, learned from train."""
    return sweep(
        "m5", "m5", [dict(rep_kw, collapse_repeats=True)],
        list(
            grid(
                learn_emissions=[True],
                p_self=[0.0, 0.2, 0.35, 0.5],
                prior=[0.5, 5.0, 20.0],
                emission_temp=[0.3, 0.5, 1.0],
                w_dur=[0.0, 1.0, 2.0],
                uni_from_scale=[0.5, 0.75],
            )
        )
        + list(  # ablation: the same HMM with a hand-set channel, nothing learned
            grid(learn_emissions=[False], p_self=[0.0, 0.2, 0.35],
                 e_self=[0.6, 0.8, 1.0], w_dur=[0.0, 1.0])
        ),
    )


def stage_m6(rep_kw):
    """Lever (c): tonic as a latent variable with a prior learned from train."""
    return sweep(
        "m6", "m6", [dict(rep_kw, collapse_repeats=True)],
        [
            {"base": b, "base_kw": dict(bk), "temperature": t, "learn_prior": lp}
            for b, bk in [
                ("m3", (("w_arohana", 1.0), ("lam_bi", 0.7), ("lam_uni", 0.1),
                        ("uni_from_scale", 0.75), ("w_dur", 1.0), ("w_skip", 0.5))),
                ("m3", (("w_arohana", 1.0), ("lam_bi", 0.7), ("lam_uni", 0.1),
                        ("uni_from_scale", 0.75), ("w_dur", 1.0), ("w_skip", 0.5),
                        ("length_norm", False))),
                ("m5", (("p_self", 0.2), ("prior", 5.0), ("emission_temp", 0.3), ("w_dur", 1.0))),
            ]
            for t in [0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
            for lp in [True, False]
        ],
    )


def stage_m7(rep_kw):
    """Everything that worked, plus per-raag hubness calibration.

    The components are pinned to the settings their own stages chose — an untuned channel
    inside a combination method measures the tuning, not the combination."""
    channel_kw = {"p_self": 0.2, "prior": 5.0, "emission_temp": 0.3, "w_dur": 1.0}
    base_kw = {"lam_bi": 0.7, "lam_uni": 0.1, "uni_from_scale": 0.75}
    return sweep(
        "m7", "m7", [dict(rep_kw, collapse_repeats=True)],
        [
            {
                "use_channel": uc,
                "channel_kw": dict(channel_kw),
                "base_kw": dict(base_kw),
                "w_crepe": wc,
                "calibrate": cal,
                "marginalise_tonic": mt,
                "temperature": 0.1,
            }
            for uc in (True, False)
            for wc in (0.0, 1.0)
            for cal in ("none", "zscore")
            for mt in (False, True)
        ],
        extra_trackers=("crepe",),
    )


def stage_m8(rep_kw):
    """Un-quantized observations: does soft swar membership beat rounding to 12 bins?

    Applied on top of the two methods worth keeping (M4 and M7), because the question is
    whether it improves the *best* pipeline, not whether it improves a toy one.
    """
    rows = []
    for name, method, mkw in [
        ("m8_m4", "m4", {"w_crepe": 1.0, "primary": "tony"}),
        ("m8_m7", "m7", {"use_channel": True,
                         "channel_kw": {"p_self": 0.2, "prior": 5.0, "emission_temp": 0.3, "w_dur": 1.0},
                         "base_kw": {"lam_bi": 0.7, "lam_uni": 0.1, "uni_from_scale": 0.75},
                         "w_crepe": 1.0, "calibrate": "zscore",
                         "marginalise_tonic": True, "temperature": 0.1}),
    ]:
        rows += sweep(
            name, method,
            [dict(rep_kw, collapse_repeats=True, quantize=q) for q in ("semitone", "shruti")],
            [mkw],
            feature_grid=list(grid(soft_sigma=[0.0, 25.0, 40.0, 55.0, 70.0],
                                   tuning_offsets=[False, True])),
            extra_trackers=("crepe",),
        )
    return rows


def stage_m9(rep_kw):
    """Un-quantized contour evidence: time-delayed melody surfaces, alone and fused."""
    # TDMS builds its surface from its own tracker's frames and therefore looks the tonic
    # up itself; it must be told the representation's policy or it silently falls back to
    # the heuristic and becomes blind to an annotated tonic.
    tm = rep_kw.get("tonic_mode", "video")
    rows = sweep(
        "m9", "m9", [dict(rep_kw, collapse_repeats=True)],
        # `metric` turned out to matter more than any of the resolution knobs: chi-square
        # compares two normalised histograms bin-by-bin relative to their own magnitude,
        # where cosine lets a few high-energy bins dominate the inner product.
        [dict(tracker=t, n_bins=nb, tau=ta, smooth=sm, tonic_mode=tm, metric=me)
         for t in ("crepe", "tony")
         for me in ("chi2", "cosine")
         for nb, ta, sm in [(40, 0.3, 1.0), (60, 0.3, 1.0), (60, 0.15, 1.0),
                            (80, 0.3, 1.0), (80, 0.15, 2.0), (120, 0.3, 2.0)]],
    )
    M4 = {"w_crepe": 1.0, "primary": "tony"}
    rows += sweep(
        "m9plus", "m9plus", [dict(rep_kw, collapse_repeats=True)],
        [{"w_tdms": w, "base": "m4", "base_kw": dict(M4),
          "tdms_kw": {"tracker": "crepe", "n_bins": nb, "tau": ta, "tonic_mode": tm,
                      "metric": me}}
         for w in (0.5, 1.0, 1.5, 2.0, 3.0)
         for me in ("chi2", "cosine")
         for nb, ta in ((80, 0.3), (60, 0.3))],
        extra_trackers=("crepe",),
    )
    return rows


def stage_m10(rep_kw):
    """Emphasis + register. Only meaningful under an annotated tonic, so that is pinned."""
    rep = dict(rep_kw, collapse_repeats=True, tonic_mode="true")
    rows = sweep(
        "m10", "m10", [rep],
        [dict(w_emph=1.0, w_reg=wr, w_vivadi=wv, vaadi_w=vw, samvaadi_w=sw,
              nyas_w=1.0, scale_w=0.5, reg_prior=0.25)
         for wr in (0.0, 0.25, 0.5, 1.0)
         for wv in (0.0, 0.5, 1.0, 2.0)
         for vw, sw in ((3.0, 2.0), (2.0, 1.5), (5.0, 2.0), (1.0, 1.0))],
    )
    # fused with the best phrase method, the same way M9 was
    M4 = {"w_crepe": 1.0, "primary": "tony"}
    rows += sweep(
        "m10plus", "m10plus", [rep],
        [{"w_reg": w, "base": "m4", "base_kw": dict(M4), "reg_kw": {}}
         for w in (0.25, 0.5, 1.0, 2.0)],
        extra_trackers=("crepe",),
    )
    return rows


def stage_m12(rep_kw):
    """The database as a prior — first on the histogram, then on the n-gram LM, then both.

    `m14` is the combination: occupancy (where the raag sits) plus transitions (how it
    moves), each with the DB mixed in as a prior rather than used as the whole model.
    """
    tm = rep_kw.get("tonic_mode", "video")
    sep = rep_kw.get("separate")
    HIST = dict(n_bins=120, source="frames", tracker="crepe", metric="chi2",
                smooth=1.0, power=0.5, tonic_mode=tm, separate=sep)
    rep = [dict(rep_kw, collapse_repeats=True)]

    rows = sweep("m12", "m12", rep,
                 [dict(HIST, lam=l, n_bins=nb)
                  for l in (0.0, 0.15, 0.3, 0.45, 0.6, 1.0)
                  for nb in (80, 120, 240)])
    rows += sweep("m13", "m13", rep,
                  [dict(lam_db=l, which=w, w_uni=wu, order=o)
                   for l in (0.0, 0.3, 0.5, 1.0)
                   for w in ("bigram_dur", "bigram", "bigram_skip")
                   for wu in (0.0, 0.3)
                   for o in (2,)])
    rows += sweep("m14", "m9plus", rep,
                  [{"w_tdms": w, "base": "m13",
                    "base_kw": dict(lam_db=0.3, which="bigram_dur", w_uni=0.3),
                    "tdms_kw": dict(HIST, lam=l), "tdms_cls": "m12"}
                   for w in (1.5, 2.0, 3.0, 4.0)
                   for l in (0.15, 0.3, 0.45)])
    return rows


def stage_m11(rep_kw):
    """The histogram floor, and what it costs to quantize it.

    `n_bins=12` is the classical pitch-class-distribution baseline; 60-120 keeps shruti.
    The gap between them is the price of rounding to semitones, measured directly.
    """
    tm = rep_kw.get("tonic_mode", "video")
    sep = rep_kw.get("separate")
    rows = sweep(
        "m11", "m11", [dict(rep_kw, collapse_repeats=True)],
        [dict(n_bins=nb, source=src, tracker=tr, metric=met, smooth=sm, power=pw,
              tonic_mode=tm, separate=sep)
         for src in ("frames",)
         for tr in ("crepe", "tony")
         for nb in (12, 24, 60, 120, 240)
         for met in ("cosine", "chi2")
         for sm, pw in ((1.0, 0.5), (2.0, 0.5), (1.0, 1.0))],
    )
    M4 = {"w_crepe": 1.0, "primary": "tony"}
    rows += sweep(
        "m11plus", "m9plus", [dict(rep_kw, collapse_repeats=True)],
        [{"w_tdms": w, "base": "m4", "base_kw": dict(M4),
          "tdms_kw": {"n_bins": 120, "tracker": "crepe", "tonic_mode": tm,
                      "separate": sep, "source": "frames"},
          "tdms_cls": "m11"}
         for w in (0.5, 1.0, 1.5, 2.0)],
        extra_trackers=("crepe",),
    )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rep", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11", "m12"])
    ap.add_argument("--rep", default=None, help="JSON dict of representation params for method stages")
    args = ap.parse_args()

    if args.stage == "rep":
        stage_rep()
    else:
        rep_kw = json.loads(args.rep) if args.rep else DEFAULT_REP
        {"m1": stage_m1, "m2": stage_m2, "m3": stage_m3, "m4": stage_m4,
         "m5": stage_m5, "m6": stage_m6, "m7": stage_m7,
         "m8": stage_m8, "m9": stage_m9, "m10": stage_m10,
         "m11": stage_m11, "m12": stage_m12}[args.stage](rep_kw)
