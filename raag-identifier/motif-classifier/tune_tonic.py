"""Choose the Sa-vs-Pa correction weights, scored by how much of each clip lands in-scale.

`chroma_tonic`'s four weights are searched here rather than in `tune.py` because the search
can be made almost free: the per-video chroma histogram and median pitch are what the
weights act on, and those don't depend on the weights at all. Compute them once, and every
candidate weighting is twelve dot products.

The objective is the fraction of sung duration falling on swars of the **true** raag —
i.e. it uses train labels, like any other hyperparameter here. It is reported with a
grouped-by-video CV split so the numbers are comparable to everything else.
"""

import itertools
import json
from pathlib import Path

import numpy as np

from diagnostics import scale_masks
from evaluate import group_folds
from extract import load_cache, list_clips
from raagdb import dataset_raags
from represent import estimate_tonic_hz, refine_tonic

RESULTS = Path(__file__).resolve().parent / "results"


def video_stats(tracker="tony", max_cents_dev=2400.0):
    """Per video: coarse tonic, chroma histogram and median pitch (semitones vs that tonic)."""
    cache = load_cache(tracker)
    meta = [c for c in list_clips() if c["clip_id"] in cache]
    by_video = {}
    for c in meta:
        by_video.setdefault(c["video"], []).append(c["clip_id"])

    stats = {}
    for video, ids in by_video.items():
        f0 = np.concatenate([cache[i]["f0"] for i in ids])
        voiced = np.concatenate([cache[i]["voiced"] for i in ids])
        t0 = estimate_tonic_hz(f0, voiced)
        f = f0[voiced & (f0 > 0)]
        cents = 1200.0 * np.log2(np.clip(f, 1e-9, None) / t0)
        cents = cents[np.abs(cents) <= max_cents_dev]
        if len(cents) < 50:
            stats[video] = (t0, np.ones(12) / 12, 6.0)
            continue
        semis = np.round(cents / 100.0).astype(int)
        h = np.bincount(semis % 12, minlength=12).astype(float)
        stats[video] = (t0, h / h.sum(), float(np.median(semis)))
    return cache, meta, stats


def pick_shift(h, median, a, b, g, mt):
    ks = np.arange(-6, 6)
    score = (
        h[ks % 12]
        + a * h[(ks + 7) % 12]
        - b * h[(ks - 7) % 12]
        - g * np.abs(median - ks - mt)
    )
    return int(ks[int(np.argmax(score))])


def objective(cache, meta, stats, masks, weights, split="train", folds=None, refine=True):
    """Mean in-scale duration fraction, and (if folds given) its per-fold spread."""
    a, b, g, mt = weights
    per_video_shift = {v: pick_shift(h, med, a, b, g, mt) for v, (_, h, med) in stats.items()}

    vals, fold_ids = [], []
    for c in meta:
        if c["split"] != split:
            continue
        entry = cache[c["clip_id"]]
        t0, _, _ = stats[c["video"]]
        tonic = t0 * 2.0 ** (per_video_shift[c["video"]] / 12.0)
        if refine:
            tonic = refine_tonic(entry["f0"], entry["voiced"], tonic)
        notes = entry["notes"]
        if len(notes) == 0:
            continue
        dur = notes[:, 1] - notes[:, 0]
        cents = 1200.0 * np.log2(np.clip(notes[:, 2], 1e-9, None) / tonic)
        keep = np.abs(cents) <= 2400.0
        if not keep.any():
            continue
        d = np.bincount(np.round(cents[keep] / 100.0).astype(int) % 12, weights=dur[keep], minlength=12)
        if d.sum() <= 0:
            continue
        vals.append(float(masks[c["raag"]] @ d / d.sum()))
        if folds is not None:
            fold_ids.append(folds[c["video"]])
    mean = float(np.mean(vals))
    if folds is None:
        return mean, 0.0
    fold_ids = np.array(fold_ids)
    vals = np.array(vals)
    per_fold = [vals[fold_ids == k].mean() for k in np.unique(fold_ids)]
    return mean, float(np.std(per_fold))


if __name__ == "__main__":
    masks = scale_masks(dataset_raags())
    cache, meta, stats = video_stats()

    class _C:  # group_folds wants objects with .raag/.video
        def __init__(self, r, v):
            self.raag, self.video = r, v

    folds = group_folds([_C(c["raag"], c["video"]) for c in meta if c["split"] == "train"])

    base, base_sd = objective(cache, meta, stats, masks, (0, 0, 0, 6), folds=folds)
    print(f"baseline (no correction, k=0 always): in_scale {base:.4f} ±{base_sd:.4f}")

    rows = []
    for a, b, g, mt in itertools.product(
        [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0, 1.5], [0.0, 0.02, 0.05, 0.1], [4.0, 6.0, 8.0]
    ):
        m, sd = objective(cache, meta, stats, masks, (a, b, g, mt), folds=folds)
        rows.append({"alpha": a, "beta": b, "gamma": g, "median_target": mt, "in_scale": m, "sd": sd})
    rows.sort(key=lambda r: -r["in_scale"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sweep_tonic.json").write_text(json.dumps(rows, indent=1))
    for r in rows[:10]:
        print(f"  in_scale {r['in_scale']:.4f} ±{r['sd']:.4f}  "
              f"alpha={r['alpha']} beta={r['beta']} gamma={r['gamma']} median_target={r['median_target']}")
