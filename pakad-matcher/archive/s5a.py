"""S5a: does the constrained-vs-free ratio rank candidates better than the absolute cost?

    poetry run python s5a.py

Evaluated on the 168 existing judgments -- no new annotation. Reported per phrase, because
ranking *within* a phrase is the thing the absolute cost could not do.
"""

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import decode
import tune


def patk(score, items, k):
    """Mean over phrases of the yes-rate among the k best-scoring candidates (lower = better)."""
    pid = np.array([it["pid"] for it in items])
    y = np.array([it["y"] for it in items])
    return float(np.mean([y[pid == p][np.argsort(score[pid == p])[:k]].mean() for p in sorted(set(pid))]))


def main():
    items = tune.spans()
    y = np.array([it["y"] for it in items])
    cost = tune.costs(items, {})
    ratio = np.array([decode.ratio(it["cents"], it["swars"], it["hop"]) for it in items])
    ratio = np.nan_to_num(ratio, nan=np.nanmax(ratio))

    rows = [("tuned cost", cost), ("ratio (constrained - free)", ratio)]
    for w in (0.5, 1.0, 2.0, 4.0):
        rows.append((f"cost + {w} x ratio", cost + w * ratio))

    print(f"{len(items)} spans, {y.mean():.2f} yes\n")
    print(f"{'score':30s} {'per-phrase AUC':>15s} {'P@1':>6s} {'P@3':>6s}")
    for name, s in rows:
        aucs = tune.per_phrase_auc(s, items)
        print(f"{name:30s} {aucs:15.3f} {patk(s, items, 1):6.2f} {patk(s, items, 3):6.2f}")

    print("\nratio, by verdict:  yes %.3f   no %.3f" % (ratio[y == 1].mean(), ratio[y == 0].mean()))
    pid = np.array([it["pid"] for it in items])
    print("\nper phrase (AUC of cost -> AUC of ratio):")
    for p in sorted(set(pid)):
        m = pid == p
        a = tune.per_phrase_auc(cost[m], [it for it in items if it["pid"] == p])
        b = tune.per_phrase_auc(ratio[m], [it for it in items if it["pid"] == p])
        print(f"  {p:22s} {a:.2f} -> {b:.2f}{'   +' if b > a + 0.02 else ('   -' if b < a - 0.02 else '')}")


if __name__ == "__main__":
    main()
