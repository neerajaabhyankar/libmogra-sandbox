"""Turn the matcher's cost into P(a listener calls this the samooha), for the tool (pakad.py).

    poetry run python calibrate.py      # -> results/calibration.json

Platt scaling of the cost under the method chosen on validation (`s7.frozen_params()`), fitted on
the VALIDATION judgments only (`audit.splits`, via `corpus.spans`); reliability is reported
leave-one-samooha-out. The chosen method's own parameters were also fitted on validation, so the
calibration sees costs from a matcher tuned on the same spans: read its Brier score as optimistic.

Until 2026-10-03 the calibration was fitted on the first 168 judgments (round 1 -- all now
validation) under the S4b matcher: kept as results/calibration_s4b.json.
Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

import _bootstrap  # noqa: F401
import config as C

MODEL_JSON = C.RESULTS_DIR / "calibration.json"
BANDS = (0.0, 0.2, 0.4, 0.6, 0.8)


def params_hash(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def fit(costs, y):
    m = LogisticRegression(max_iter=1000).fit(np.asarray(costs).reshape(-1, 1), np.asarray(y))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def probability(cost, model=None):
    if model is None:
        model = json.loads(MODEL_JSON.read_text())
    return 1.0 / (1.0 + np.exp(-(model["w"] * np.asarray(cost) + model["b"])))


def main():
    import corpus
    import s7
    params = s7.frozen_params()
    items = corpus.spans("validation")
    y = np.array([it["y"] for it in items])
    pid = np.array([it["pid"] for it in items])
    cost = s7.match_cost(items, params)
    pred = np.zeros(len(y))
    for p in sorted(set(pid)):
        tr = pid != p
        w, b = fit(cost[tr], y[tr])
        pred[~tr] = probability(cost[~tr], dict(w=w, b=b))
    brier = float(np.mean((pred - y) ** 2))
    print(f"validation, leave-one-samooha-out Brier {brier:.3f}  "
          f"(always-guess-the-base-rate {np.mean((y.mean() - y) ** 2):.3f})")
    for lo in BANDS:
        m = (pred >= lo) & (pred < lo + 0.2)
        if m.sum():
            print(f"  p in [{lo:.1f},{lo + 0.2:.1f})  n={m.sum():3d}  actual yes-rate {y[m].mean():.2f}")
    w, b = fit(cost, y)
    MODEL_JSON.write_text(json.dumps(dict(w=w, b=b, n=len(y), brier_lopo=brier, fitted_on="validation",
                                          method=json.loads(s7.CHOICE_JSON.read_text())["chosen"],
                                          params_hash=params_hash(params)), indent=1))
    print(f"fitted on all {len(y)} validation spans: p = sigmoid({w:.2f} * cost + {b:.2f}) -> {MODEL_JSON}")


if __name__ == "__main__":
    main()
