"""Final numbers: best-on-train config for each method, run once on the held-out test split.

Reads the winning config straight out of `results/sweep_<method>.json` so the reported
numbers cannot drift from what tuning actually chose. Writes `results/final.json` and
`results/RESULTS.md`.

    poetry run python report.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate import evaluate, evaluate_by_video, make_method
from features import build_features
from confusion import plot_confusion
from musical_eval import calibrate_temperature, musical_metrics, worst_and_best_mistakes
from raagdb import dataset_raags
from represent import Params, build_clips

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

METHODS = [
    ("m1", "exact mukhyanga substring match"),
    ("m2", "n-gram / skip-gram phrase overlap + scale term"),
    ("m3", "phrase grammar, smoothed bigram log-likelihood"),
    ("m4", "M3 grammar over Tony **+ CREPE** fused"),
    ("m5", "noisy-channel HMM, kan-swar/meend emissions learned from train"),
    ("m6", "joint tonic + raag, rotation prior learned from train"),
    ("m7", "channel + CREPE + tonic prior + per-raag calibration"),
    ("m9", "time-delayed melody surface — un-quantized contour, **no** mukhyanga"),
    ("m9plus", "M4 + melody surface (quantized phrases + continuous contour)"),
]


# Re-ranking budget per method. M6/M7 score all 12 rotations and refit per fold, so they
# get a shorter shortlist and fewer seeds — the shortlist is already tightly clustered.
RERANK = {"m6": (8, (0, 1, 2)), "m7": (6, (0, 1, 2)),
          "m9": (6, (0, 1, 2)), "m9plus": (4, (0, 1, 2))}


def best_config(method, shortlist=20, seeds=(0, 1, 2, 3, 4)):
    """Re-rank the sweep's top configs over several CV fold seeds before committing.

    A sweep over thousands of configs with a per-fold spread of ~0.03 will hand back a
    winner that is partly just a lucky fold assignment. Re-scoring the shortlist against
    several independent groupings and taking the best *average* costs seconds and picks a
    config that is actually robust — still entirely on train.
    """
    from tune import cv_score

    shortlist, seeds = RERANK.get(method, (shortlist, seeds))
    rows = json.loads((RESULTS / f"sweep_{method}.json").read_text())
    rows.sort(key=lambda r: -r["top1"])
    best, best_mean = None, -1.0
    for r in rows[:shortlist]:
        scores = [
            cv_score(method, Params(**r["rep"]), r["method"], feature_kw=r.get("features"),
                     seed=sd, extra_trackers=tuple(r.get("extra_trackers", ())))["top1"]
            for sd in seeds
        ]
        mean = float(np.mean(scores))
        if mean > best_mean:
            best, best_mean = r, mean
    best = dict(best)
    best["top1_single_seed"] = best["top1"]
    best["top1"] = best_mean
    best["top1_std"] = float(
        np.std([
            cv_score(method, Params(**best["rep"]), best["method"], feature_kw=best.get("features"),
                     seed=sd, extra_trackers=tuple(best.get("extra_trackers", ())))["top1"]
            for sd in seeds
        ])
    )
    return best


def run_split(method, cfg, split):
    """Score `split` with `cfg`. Anything the method learns is fit on **train only**."""
    from tune import split_feats

    rep = Params(**cfg["rep"])
    extra = tuple(cfg.get("extra_trackers", ()))
    feats = split_feats(rep, cfg.get("features", {}) or {}, split, extra)
    m = make_method(method, **cfg["method"])
    if m.fitted:
        m.fit(split_feats(rep, cfg.get("features", {}) or {}, "train", extra))
    metrics, rows = evaluate(m, feats, keep_scores=True)
    metrics["video_top1"], metrics["n_videos"] = evaluate_by_video(rows, m.raags)
    return metrics, rows


def main(from_cache=False):
    cached = json.loads((RESULTS / "final.json").read_text()) if from_cache else None
    labels = sorted(dataset_raags())
    out = {}
    mus_lines = []
    lines = [
        "# Results",
        "",
        "50-way raag identification on `hindustani-raag-small`. Chance = 0.020 (1/50); the",
        "majority class is 3.6 % of test. Train numbers are 5-fold **grouped-by-video** CV on the",
        "train split (used for tuning); test numbers are a single pass over the held-out test",
        "split with the config that CV chose, run once.",
        "",
        "| method | train top-1 (CV) | **test top-1** | test top-5 | test MRR | test top-1 by video-vote | confusion matrix |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, blurb in METHODS:
        if cached and method in cached:
            cfg = cached[method]["config"]
            test_m, test_rows = dict(cached[method]["test"]), cached[method]["test_rows"]
            T = cached[method]["musical"]["temperature"]
            test_m["video_top1"], test_m["n_videos"] = evaluate_by_video(test_rows, labels)
        else:
            cfg = best_config(method)
            test_m, test_rows = run_split(method, cfg, "test")
            # temperature is calibrated on TRAIN, then applied unchanged to the test scores
            _, train_rows = run_split(method, cfg, "train")
            T = calibrate_temperature(train_rows, labels)
        mus = musical_metrics(test_rows, temperature=T)
        test_m.update({f"mus_{k}": v for k, v in mus.items()})
        png = plot_confusion(
            test_rows,
            labels,
            f"{method} — {blurb}".replace("**", ""),  # titles are plain text, not markdown
            f"test split, 92 clips over 50 candidate raags · "
            f"top-1 {test_m['top1']:.3f} · top-5 {test_m['top5']:.3f} · "
            f"diagonal outlined = correct",
            RESULTS / f"confusion_{method}.png",
        )
        out[method] = {"config": cfg, "test": test_m, "confusion_png": png.name,
                       "musical": mus}
        mus_lines.append(
            f"| **{method}** | {mus['mistake_affinity']:.3f} | {mus['mistake_affinity_chance']:.3f} | "
            f"{mus['expected_affinity']:.3f} | {mus['expected_affinity_chance']:.3f} | "
            f"{mus['affinity_ce']:.3f} | {mus['nll']:.3f} | "
            f"{mus['tonic_explained']:.3f} | {mus['tonic_explained_chance']:.3f} | "
            f"{mus['rot_affinity']:.3f} | {mus['rot_affinity_chance']:.3f} |"
        )
        lines.append(
            f"| **{method}** — {blurb} | {cfg['top1']:.3f} ± {cfg['top1_std']:.3f} | "
            f"**{test_m['top1']:.3f}** ({round(test_m['top1']*test_m['n_clips'])}/{test_m['n_clips']}) | "
            f"{test_m['top5']:.3f} | {test_m['mrr']:.3f} | {test_m['video_top1']:.3f} | "
            f"[`{png.name}`]({png.name}) |"
        )
        out[method]["test_rows"] = test_rows

    lines += [
        "| _chance_ | 0.020 | 0.020 | 0.100 | 0.090 | 0.020 | — |",
        "",
        "## How bad are the mistakes?",
        "",
        "Accuracy scores Tilak Kamod → Des (a near-miss any listener could make) exactly as",
        "badly as Tilak Kamod → Bairagi (nothing in common). These grade against",
        "`raagspace.affinity()`, a raag-to-raag similarity built only from the libmogra",
        "database — TF-IDF over mukhyanga/aaroha n-grams, swar-set Jaccard, and thaat.",
        "Each metric sits next to the value random guessing would score.",
        "",
        "- **mistake affinity** — mean affinity(true, predicted) over errors only. Higher = misses land nearby.",
        "- **expected affinity (MEA)** — `Σ_r p(r)·affinity(true,r)` over the whole softmax output, not just the argmax.",
        "- **affinity CE** — cross-entropy against a soft target `q ∝ affinity(true,·)^4`. **Lower is better.** This is the mukhyanga-based loss: it does not punish mass on genuinely related raags.",
        "- **NLL** — ordinary negative log-likelihood of the true raag, for reference (chance = ln 50 = 3.912).",
        "- **tonic-explained** — of the errors, the share whose prediction is a near-exact *rotation* of the true scale. Those are Sa-placement failures, not raag failures.",
        "",
        "Softmax temperature is calibrated per method on **train** (standard temperature",
        "scaling, minimising NLL) so methods with different score scales are comparable.",
        "",
        "| method | mistake affinity | (chance) | MEA | (chance) | affinity CE ↓ | NLL ↓ | tonic-explained | (chance) | rot-affinity | (chance) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ] + mus_lines + [
        "",
        "## Chosen configurations",
        "",
    ]
    for method, _ in METHODS:
        cfg = out[method]["config"]
        lines += [
            f"**{method}**",
            "```",
            f"representation: {cfg['rep']}",
            f"method:         {cfg['method']}",
            f"features:       {cfg.get('features', {})}",
            "```",
            "",
        ]

    # what the best method actually gets right, and what it confuses
    best = max(METHODS, key=lambda mb: out[mb[0]]["test"]["top1"])[0]
    rows = out[best]["test_rows"]
    correct = sorted({r["true"] for r in rows if r["rank"] == 1})
    pred_counts = Counter(r["pred"] for r in rows).most_common(6)
    near, far = worst_and_best_mistakes(rows)
    lines += [
        f"## Error structure ({best}, test)",
        "",
        f"- correct at rank 1: {', '.join(correct) if correct else '(none)'}",
        f"- median rank of the true raag: {np.median([r['rank'] for r in rows]):.0f} of 50",
        f"- most-predicted labels: {', '.join(f'{p} ({c})' for p, c in pred_counts)}",
        "",
        "Most defensible misses (highest affinity):",
        "",
    ] + [f"  - {t} → {pr}  (affinity {a:.2f})" for a, t, pr, _, _ in near] + [
        "",
        "Least defensible misses (lowest affinity); `rot` is the affinity after rotating the",
        "predicted scale by `k` semitones — a high `rot` means this was a tonic error:",
        "",
    ] + [f"  - {t} → {pr}  (affinity {a:.2f}, rot {ar:.2f} at k={k})" for a, t, pr, ar, k in far] + [""]

    (RESULTS / "RESULTS.md").write_text("\n".join(lines))
    (RESULTS / "final.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    import sys

    main(from_cache="--metrics-only" in sys.argv)
