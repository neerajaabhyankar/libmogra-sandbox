"""Score the held-out 150 for runs that already finished, without retraining anything.

Every run saves `best.pt` and the exact `config` it was launched with, so a test score is
inference over 150 clips — seconds, not hours. This rebuilds each model from its own
recorded config, loads its checkpoint, scores test, and writes the result back into that
run's `result.json` under `"test"`.

    poetry run python scripts/91_score_test.py                # every run missing a test score
    poetry run python scripts/91_score_test.py c4h c2         # just these
    poetry run python scripts/91_score_test.py --force        # rescore even if present
    poetry run python scripts/91_score_test.py --device cpu   # leave the GPU to a live batch

Runs launched from now on score test themselves at the end (`10_train.py`, `--no-test` to
opt out), so this is for the backlog.

**What a test number here does and does not license.** The training loop never sees test:
splits are video-disjoint (362 train-pool videos, 50 test videos, zero overlap) and
checkpoint selection is on val top-1. So each individual number is honest. What is *not*
honest is picking the best of N test scores and reporting it as the method's performance —
with 150 clips the standard error on top-1 is around 4 points, so the maximum over a dozen
methods is optimistically biased by roughly that much. Choose the method on val; read test
as the reported result for a method chosen some other way.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from common import metrics, trainer
from common.data import load_clips
from common.paths import RESULTS
from common.tonic import shuffled_tonics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_train = import_module("10_train")


class _Args:
    """The recorded config, back as the attribute bag the helpers in 10_train expect."""

    def __init__(self, cfg):
        self.__dict__.update(cfg)


def score_one(run_dir, device="auto", force=False):
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text())
    if "test" in result and not force:
        return None, "already scored"
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        return None, "no best.pt"

    args = _Args(result["config"])
    args.device = device
    cfg = trainer.TrainConfig(batch_size=getattr(args, "batch_size", 16),
                              num_workers=0, device=device,
                              seed=getattr(args, "seed", 0))

    model = _train.build_model(args)
    model.load_state_dict(torch.load(ckpt, map_location=cfg.device, weights_only=True))
    model.to(cfg.device)

    override = (shuffled_tonics(load_clips(), seed=args.seed)
                if getattr(args, "shuffle_tonics", False) else None)
    test_clips = load_clips("test")
    ds = _train.make_dataset(args.arch, test_clips, args, override, train=False)
    logits = trainer.predict(model, ds, cfg)
    te_m, te_rows = metrics.score(test_clips, logits)

    # reuse the temperature fitted on val, exactly as a live run would
    T = result.get("temperature", 1.0)
    te_mus = metrics.musical(te_rows, temperature=T)

    result["test"] = {"metrics": te_m, "musical": te_mus, "scored_by": "91_score_test.py"}
    result_path.write_text(json.dumps(result, indent=2, default=str))
    np.save(run_dir / "test_logits.npy", logits)
    metrics.confusion(te_rows, run_dir / "confusion_test.png",
                      title=f"{run_dir.name} test — top-1 {te_m['top1']:.3f}")
    return te_m, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run ids; default every run with a result.json")
    ap.add_argument("--force", action="store_true", help="rescore even if already present")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dirs = ([RESULTS / r for r in a.runs] if a.runs
            else sorted(d for d in RESULTS.iterdir()
                        if (d / "result.json").exists() and (d / "best.pt").exists()))

    print(f"{'run':<14} {'test top-1':>10} {'top-5':>7} {'video':>7}   note")
    scored = []
    for d in dirs:
        if not (d / "result.json").exists():
            print(f"{d.name:<14} {'-':>10}                     no result.json")
            continue
        try:
            m, note = score_one(d, device=a.device, force=a.force)
        except Exception as e:                       # one bad run must not stop the sweep
            print(f"{d.name:<14} {'-':>10}                     FAILED {type(e).__name__}: {e}")
            continue
        if m is None:
            print(f"{d.name:<14} {'-':>10}                     {note}")
            continue
        print(f"{d.name:<14} {m['top1']:>10.3f} {m['top5']:>7.3f} {m['video_vote']:>7.3f}   scored")
        scored.append((m["top1"], d.name))

    # The leaderboard spans every run that HAS a test score, not just the ones scored in
    # this pass -- a re-run that skips the best model must not report a worse one as the
    # leader. Rendered by common.report like every other table in the project.
    from common.report import load_runs, table

    all_runs = load_runs()
    with_test = [r for r in all_runs if r.get("test")]
    if scored:
        print(f"\n{len(scored)} run(s) scored this pass.")
    if with_test:
        ranked = sorted(with_test,
                        key=lambda r: -r["test"]["metrics"]["top1"])
        print(f"\n{len(with_test)} run(s) have a test score:\n")
        print(table("notebook", all_runs, order=[r["run_id"] for r in ranked]))
        best = ranked[0]
        print(f"\nHighest: {best['run_id']} at {best['test']['metrics']['top1']:.3f} -- "
              f"optimistically biased as a maximum, see this file's docstring. "
              f"Pick methods on val.")


if __name__ == "__main__":
    main()
