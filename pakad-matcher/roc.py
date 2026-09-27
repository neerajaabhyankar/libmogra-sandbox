"""ROC curves for test1 (is this the samooha?) and test2 (is this swar used in aaroh / avaroh?).

    poetry run python roc.py            # -> results/roc/test1.png, validation.png, test2.png

Test1 scores are not comparable across samoohas (a cost for `m D n D` means nothing next to one
for `g S r S`), so each span's score is replaced by its rank within its samooha before pooling.
The legend gives per-samooha AUC -- the number every table in plan.md reports.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
import config as C
import corpus

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
                label=f"{name.split(' (')[0]}  {corpus.per_samooha_auc(s, items):.3f}"
                      + ("  (chosen)" if bold else ""))
    _axes(ax, f"{split}: is this the samooha?  {len(items)} spans, "
              f"{len({it['pid'] for it in items})} samoohas")
    ax.legend(title="per-samooha AUC", frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"{'test1' if split == 'test' else split}.png", dpi=130)


def test2():
    rows = json.loads((C.RESULTS_DIR / "test2.json").read_text())
    methods = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw=dict(width_ratios=[1, 1.1]))
    for m, col in zip(methods, COLOURS):
        s = [r[m]["up_frac"] for r in rows] + [1 - r[m]["up_frac"] for r in rows]
        y = [int(r["aaroh"]) for r in rows] + [int(r["avaroh"]) for r in rows]
        fpr, tpr = roc(s, y)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{m}  {auc:.3f}")
    _axes(ax, f"test2: used in aaroh? used in avaroh?  {2 * len(rows)} questions")
    ax.legend(title="AUC", frameon=False, fontsize=8, loc="lower right")

    # the evidence behind it: fraction of approaches from below, per swar
    kind = lambda r: "aaroh only" if not r["avaroh"] else ("avaroh only" if not r["aaroh"] else "both")
    marker = {"aaroh only": "^", "avaroh only": "v", "both": "o"}
    labels = [f"{r['raag']} {r['swar']}" for r in rows]
    for i, r in enumerate(rows):
        for j, (m, col) in enumerate(zip(methods, COLOURS)):
            bx.scatter(r[m]["up_frac"], i + 0.18 * (j - 0.5), color=col, marker=marker[kind(r)], s=36,
                       label=m if i == 0 else None)
    bx.set_yticks(range(len(rows)), labels, fontsize=8)
    bx.invert_yaxis()
    bx.axvline(0.5, color="#bbbbbb", lw=0.8, ls=":")
    bx.set(xlim=(0, 1), xlabel="fraction of occurrences approached from below",
           title="per swar:  ▲ aaroh only   ▼ avaroh only   ● both")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "test2.png", dpi=130)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        test1(split)
    if (C.RESULTS_DIR / "test2.json").exists():
        test2()
    print(f"-> {OUT}")
