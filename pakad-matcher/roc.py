"""ROC curves for test1 (is this the samooha?) and test2 (is this swar used in aaroh / avaroh?).

    poetry run python roc.py            # -> results/roc/test1.png, validation.png, test2.png

Test1 scores are not comparable across samoohas (a cost for `m D n D` means nothing next to one
for `g S r S`), so each span's score is replaced by its rank within its samooha before pooling.
The legend gives per-samooha AUC -- the number every table in plan.md reports.

Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
import config as C
import corpus
import metrics

OUT = C.RESULTS_DIR / "roc"
COLOURS = ["#5b6c8f", "#b0706a", "#6f9a7a", "#8c7aa8", "#c29a5b", "#7a8c8c"]


def roc(score, y):
    order = np.argsort(-np.asarray(score), kind="stable")
    y = np.asarray(y)[order]
    tpr = np.r_[0, np.cumsum(y) / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(1 - y) / max((1 - y).sum(), 1)]
    return fpr, tpr


def within_rank(score, items):
    """Each score as its rank percentile inside its own samooha."""
    score, pid = np.asarray(score, float), np.array([it["pid"] for it in items])
    out = np.zeros_like(score)
    for p in set(pid):
        m = pid == p
        out[m] = (np.argsort(np.argsort(score[m])) + 0.5) / m.sum()
    return out


def _axes(ax, title):
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8, ls=":")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="false positive rate", ylabel="true positive rate",
           title=title, aspect="equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def test1(split):
    import s7
    choice = json.loads(s7.CHOICE_JSON.read_text())
    items = corpus.spans(split)
    sc = s7.scores(items, choice)
    y = [it["y"] for it in items]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for (name, s), col in zip(sc.items(), COLOURS):
        fpr, tpr = roc(within_rank(s, items), y)
        bold = name == choice["chosen"].replace(" (leave-one-samooha-out)", "")
        ax.plot(fpr, tpr, color=col, lw=2.2 if bold else 1.2,
                label=f"{name.split(' (')[0]}  {metrics.per_samooha_auc(s, items):.3f}"
                      + ("  (chosen)" if bold else ""))
    _axes(ax, f"{'test1' if split == 'test' else split}: is this the samooha?  {len(items)} spans, "
              f"{len({it['pid'] for it in items})} samoohas")
    ax.legend(title="per-samooha AUC", frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"{'test1' if split == 'test' else split}.png", dpi=130)


KIND_COLOUR = {"aaroh only": "#b0706a", "avaroh only": "#5b6c8f", "both": "#9a9a9a"}
KIND_MARKER = {"aaroh only": "^", "avaroh only": "v", "both": "o"}


def _kind(r):
    return "aaroh only" if not r["avaroh"] else ("avaroh only" if not r["aaroh"] else "both")


def test2():
    """Two figures: the ROC, and per swar the up-fraction each method measured."""
    d = json.loads((C.RESULTS_DIR / "test2.json").read_text())
    rows = d["rows"] if isinstance(d, dict) else d
    methods = [k for k in rows[0] if isinstance(rows[0][k], dict)]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for m, col in zip(methods, COLOURS):
        s = [r[m]["up_frac"] for r in rows] + [1 - r[m]["up_frac"] for r in rows]
        y = [int(r["aaroh"]) for r in rows] + [int(r["avaroh"]) for r in rows]
        fpr, tpr = roc(s, y)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{m}  {np.trapz(tpr, fpr):.3f}")
    _axes(ax, f"test2: used in aaroh? used in avaroh?  {2 * len(rows)} questions")
    ax.legend(title="AUC", frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "test2_roc.png", dpi=130)

    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 7.5), sharey=True)
    labels = [f"{r['raag']} {r['swar']}" for r in rows]
    for bx, m in zip(np.atleast_1d(axes), methods):
        for i, r in enumerate(rows):
            k = _kind(r)
            bx.scatter(r[m]["up_frac"], i, color=KIND_COLOUR[k], marker=KIND_MARKER[k], s=55)
        bx.axvline(0.5, color="#bbbbbb", lw=0.8, ls=":")
        bx.set(xlim=(0, 1), title=m, xlabel="up-fraction: share of occurrences\nfollowed by a higher note")
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(rows)), labels, fontsize=8)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=KIND_COLOUR[k], marker=KIND_MARKER[k], ls="", ms=8, label=k)
               for k in KIND_COLOUR]
    fig.legend(handles=handles, title="Neeraja's label", frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "test2_scatter.png", dpi=130)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        test1(split)
    if (C.RESULTS_DIR / "test2.json").exists():
        test2()
    print(f"-> {OUT}")
