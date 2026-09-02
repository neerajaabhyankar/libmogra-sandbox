"""Scoring. Accuracy is the headline; it is not the interesting part.

Top-1 over 50 classes treats "Tilak Kamod called Des" (same thaat, same material, a mistake
a listener makes) exactly like "Tilak Kamod called Bairagi" (nothing in common). So every
run here reports two families of number:

    the usual        top-1, top-5, MRR, macro-F1, and a per-video vote
    the graded       mistake affinity, expected affinity, affinity cross-entropy, and how
                     many errors are really *tonic* errors -- the true raag transposed

The graded metrics come from utils/musical_eval.py unchanged, so a number
here is directly comparable to a number there. They are built only from the libmogra
database, never fitted, so no method gets an advantage from them.

    m, rows = score(clips, logits)          # logits: (n_clips, 50)
    m |= musical(rows, temperature=T)
"""

import re
from collections import defaultdict

import numpy as np

from .data import labels as dataset_labels
from .paths import add_sibling_paths


def score(clips, logits, top_k=(1, 5), keep_scores=True):
    """Standard metrics + one row per clip, in the shape musical_eval expects."""
    L = dataset_labels()
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (len(clips), len(L)):
        raise ValueError(f"logits {logits.shape} != ({len(clips)}, {len(L)})")

    order = np.argsort(-logits, axis=1)
    ranks, rows, hits = [], [], {k: 0 for k in top_k}
    for i, c in enumerate(clips):
        rank = int(np.where(order[i] == c.label)[0][0]) + 1
        ranks.append(rank)
        for k in top_k:
            hits[k] += rank <= k
        rows.append({
            "clip_id": c.clip_id,
            "video": c.video,
            "true": c.raag,
            "pred": L[int(order[i, 0])],
            "rank": rank,
            **({"scores": [float(x) for x in logits[i]]} if keep_scores else {}),
        })

    n = max(len(clips), 1)
    y_true = np.array([c.label for c in clips])
    y_pred = order[:, 0]
    vote, n_videos = video_vote(rows, L)
    return {
        f"top{k}": hits[k] / n for k in top_k
    } | {
        "mrr": float(np.mean([1.0 / r for r in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "macro_f1": _macro_f1(y_true, y_pred, len(L)),
        "video_vote": vote,
        "n_clips": len(clips),
        "n_videos": n_videos,
    }, rows


def _macro_f1(y_true, y_pred, n_classes):
    """Unweighted mean F1 over classes present in `y_true`. Matters because the corpus is
    unbalanced (18-73 clips per raag) and accuracy quietly rewards predicting the big ones."""
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def video_vote(rows, label_names=None):
    """One verdict per recording, by summing the chunks' score vectors.

    Pooling scores rather than majority-voting is deliberate: test videos contribute 3
    chunks, a 1-1-1 disagreement has no majority, and breaking the tie by name made the
    metric depend on the process hash seed in the sibling project.
    """
    L = label_names or dataset_labels()
    by_video = defaultdict(list)
    for r in rows:
        v = r.get("video") or re.search(r"\[(.+)\]", r["clip_id"]).group(1)
        by_video[(v, r["true"])].append(r)
    correct = 0
    for (_v, true), rs in by_video.items():
        if all("scores" in r for r in rs):
            pooled = np.sum([np.asarray(r["scores"], float) for r in rs], axis=0)
            vote = L[int(np.argmax(pooled))]
        else:
            preds = [r["pred"] for r in rs]
            vote = max(sorted(set(preds)), key=preds.count)
        correct += vote == true
    return correct / max(len(by_video), 1), len(by_video)


def calibrate_temperature(rows, grid=None):
    """Softmax temperature fit on these rows by minimising NLL. Fit on train/val, then
    applied unchanged elsewhere -- the graded metrics compare probability *shapes*, so
    methods must be put on a common footing first."""
    from utils.musical_eval import calibrate_temperature as _cal

    return _cal(rows, dataset_labels(), grid=grid)


def musical(rows, temperature=1.0, **kw):
    """The graded metrics -- see `utils/musical_eval.py` for what each means."""
    from utils.musical_eval import musical_metrics

    return musical_metrics(rows, temperature=temperature, dataset_labels=dataset_labels(),
                           **kw)


def confusion(rows, out_path, title="", label_names=None):
    """50x50 confusion matrix as a PNG. Big, but it is the artifact that tells you *what*
    the model is doing -- which raags are solved, which are hubs absorbing everything."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = label_names or dataset_labels()
    idx = {r: i for i, r in enumerate(L)}
    cm = np.zeros((len(L), len(L)), dtype=int)
    for r in rows:
        cm[idx[r["true"]], idx[r["pred"]]] += 1

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(L)):
        for j in range(len(L)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks(range(len(L)), L, rotation=90, fontsize=7)
    ax.set_yticks(range(len(L)), L, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def summary_line(m):
    """One line for a log or a table row."""
    return (f"top1 {m['top1']:.3f} | top5 {m['top5']:.3f} | mrr {m['mrr']:.3f} | "
            f"macroF1 {m['macro_f1']:.3f} | video {m['video_vote']:.3f} "
            f"({m['n_clips']} clips)")
