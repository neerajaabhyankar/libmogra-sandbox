"""The control the Stage 5 hybrid needs: the melody histogram *alone*, on the same split.

`hybrid_feat` puts two things in one model -- a learned Sa-anchored CQT and a hand-built
pitch histogram. A number for the pair is uninterpretable without a number for each part
measured the same way, and the CQT half already has one (`c4h`, `aug_jitter`). This is the
other half: multinomial logistic regression on the 120-bin histogram, fitted on exactly the
clips the DL models were fitted on, scored on exactly their val and test clips.

No network, no epochs -- seconds. The classifier is `01_probe_representations._logreg_scores`
itself rather than a second copy of it, so "melody alone" means the same thing here as it
does in the probe table.

    poetry run python scripts/21_melody_only.py                 # writes results/v1.1/melody_only/
    poetry run python scripts/21_melody_only.py --seed 1        # the other split, for the seed spread
"""

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from common import melody, metrics
from common.data import grouped_split, labels, load_clips, summarise
from common.paths import DATA_REVISION, RESULTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
_probe = import_module("01_probe_representations")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="melody_only")
    ap.add_argument("--seed", type=int, default=0, help="must match the DL run's seed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fit, val = grouped_split(load_clips("train"), val_frac=0.2, seed=a.seed)
    test = load_clips("test")
    print(f"=== {a.run_id} | melody histogram + logreg | seed {a.seed} ===")
    print(f"  fit on {summarise(fit)}")
    print(f"  score {len(val)} val, {len(test)} test")
    if a.dry_run:
        return

    t0 = time.time()
    n = len(labels())
    y = np.array([c.label for c in fit])
    X = melody.cached(fit)
    scores = {"val": _probe._logreg_scores(X, y, melody.cached(val), n),
              "test": _probe._logreg_scores(X, y, melody.cached(test), n)}

    val_m, val_rows = metrics.score(val, scores["val"])
    T = metrics.calibrate_temperature(val_rows)
    te_m, te_rows = metrics.score(test, scores["test"])
    print(f"\n  val  {metrics.summary_line(val_m)}")
    print(f"  TEST {metrics.summary_line(te_m)}")

    out_dir = RESULTS / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({
        "run_id": a.run_id, "stage": 5, "arch": "melody",
        "data_revision": DATA_REVISION,
        "config": {"feature": "melody_hist", **melody.DEFAULTS, "clf": "logreg",
                   "seed": a.seed, "arch": "melody"},
        "split": "grouped-val", "metrics": val_m,
        "musical": metrics.musical(val_rows, temperature=T), "temperature": T,
        "test": {"metrics": te_m, "musical": metrics.musical(te_rows, temperature=T)},
        "best_epoch": None, "wall_clock_s": round(time.time() - t0, 1),
    }, indent=2, default=str))
    np.save(out_dir / "test_logits.npy", scores["test"])
    metrics.confusion(te_rows, out_dir / "confusion_test.png",
                      title=f"{a.run_id} test — top-1 {te_m['top1']:.3f}")
    print(f"\nwrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
