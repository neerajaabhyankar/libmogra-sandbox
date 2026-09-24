"""Turn the matcher's cost into P(a listener calls this the phrase).

A downstream tool needs a number it can threshold, not a cost whose meaning shifts per phrase.
Platt scaling on the 168 judgments, validated leave-one-phrase-out (so the reported reliability
is for a samooha the fit never saw).

    poetry run python calibrate.py
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

import _bootstrap  # noqa: F401
import config as C
import tune

MODEL_JSON = C.RESULTS_DIR / "calibration.json"


def fit(costs, y):
    m = LogisticRegression(max_iter=1000).fit(np.asarray(costs).reshape(-1, 1), np.asarray(y))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def probability(cost, model=None):
    if model is None:
        model = json.loads(MODEL_JSON.read_text())
    return 1.0 / (1.0 + np.exp(-(model["w"] * np.asarray(cost) + model["b"])))


def main():
    items = tune.spans()
    y = np.array([it["y"] for it in items])
    pid = np.array([it["pid"] for it in items])
    cost = tune.costs(items, {})

    pred = np.zeros(len(y))
    for p in sorted(set(pid)):
        tr = pid != p
        w, b = fit(cost[tr], y[tr])
        pred[~tr] = probability(cost[~tr], dict(w=w, b=b))
    brier = float(np.mean((pred - y) ** 2))
    print(f"leave-one-phrase-out Brier {brier:.3f}  (always-guess-the-base-rate "
          f"{np.mean((y.mean() - y) ** 2):.3f})")
    print("\nreliability: of the candidates in each predicted band, how many were a yes")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        m = (pred >= lo) & (pred < lo + 0.2)
        if m.sum():
            print(f"  p in [{lo:.1f},{lo + 0.2:.1f})  n={m.sum():3d}  actual yes-rate {y[m].mean():.2f}")

    w, b = fit(cost, y)
    MODEL_JSON.parent.mkdir(exist_ok=True)
    MODEL_JSON.write_text(json.dumps(dict(w=w, b=b, n=len(y), brier_lopo=brier,
                                          matcher=C.MATCHER_VERSION, pool=C.POOL_VERSION), indent=1))
    print(f"\nfitted on all {len(y)}: p = sigmoid({w:.2f} * cost + {b:.2f}) -> {MODEL_JSON}")
    for c in (0.0, 0.25, 0.5, 1.0, 2.0):
        print(f"  cost {c:.2f} -> p {float(probability(c, dict(w=w, b=b))):.2f}")


if __name__ == "__main__":
    main()
