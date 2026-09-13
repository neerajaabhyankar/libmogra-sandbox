"""Score finished runs on many windows of the full val/test recordings, not just the Hub clips.

The Hub keeps five clips of each validation video and three of each test video. The full
recordings offer hundreds of 20 s windows. Scoring a model on `--windows` fixed, evenly
spaced windows per video changes nothing about *which* videos are evaluated -- still the
same 92 val and 50 test, from the splits file -- but measures each one far more thoroughly,
so a video's verdict no longer rests on whichever minute and a half the Hub happened to keep.

    poetry run python scripts/92_score_full.py aug_jitter full_aug      # these runs
    poetry run python scripts/92_score_full.py --windows 20 --force     # rescore

Hub-trained and full-trained runs are scored identically, so this is the column that
compares them on neutral ground. Writes into each run's `result.json`:

    "full_eval": {"windows_per_video": 20,
                  "val":  {metrics over 92 x 20 windows; video_vote pools each video's 20},
                  "test": {... 50 x 20 ...}}

The windows use the same usable-position filter as full-audio training (`--trim-seconds`,
`--loud-fraction`), so evaluation skips the dead air that training skips.
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from common import fullaudio, metrics, trainer
from common.paths import RESULTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
_train = import_module("10_train")


class _Args:
    def __init__(self, cfg):
        self.__dict__.update(cfg)


def score_run(run_dir, windows, device, force=False, trim_seconds=30.0, loud_fraction=0.8):
    path = run_dir / "result.json"
    result = json.loads(path.read_text())
    if "full_eval" in result and not force:
        return result["full_eval"], "already scored"
    if result.get("arch") != "cqt" or not (run_dir / "best.pt").exists():
        return None, "not a single-split cqt run with a best.pt"

    args = _Args(result["config"])
    cfg = trainer.TrainConfig(batch_size=16, num_workers=0, device=device,
                              seed=getattr(args, "seed", 0))
    model = _train.build_model(args)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=cfg.device,
                                     weights_only=True))
    model.to(cfg.device)

    out = {"windows_per_video": windows}
    for role in ("val", "test"):
        ds = fullaudio.FullAudioCQTDataset(fullaudio.videos(role, getattr(args, "seed", 0)),
                                           windows, train=False, trim_seconds=trim_seconds,
                                           loud_fraction=loud_fraction)
        m, _rows = metrics.score(ds.clips, trainer.predict(model, ds, cfg))
        out[role] = m
    result["full_eval"] = out
    path.write_text(json.dumps(result, indent=2, default=str))
    return out, "scored"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run ids under results/v1.1/")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    print(f"{'run':<18} {'val clip':>8} {'val video':>9} {'test clip':>9} {'test video':>10}")
    for run in a.runs:
        try:
            fe, note = score_run(RESULTS / run, a.windows, a.device, a.force)
        except Exception as e:                        # one bad run must not stop the rest
            print(f"{run:<18} FAILED {type(e).__name__}: {e}")
            continue
        if fe is None:
            print(f"{run:<18} skipped: {note}")
            continue
        v, t = fe["val"], fe["test"]
        print(f"{run:<18} {v['top1']:>8.3f} {v['video_vote']:>9.3f} {t['top1']:>9.3f} "
              f"{t['video_vote']:>10.3f}   {note}")


if __name__ == "__main__":
    main()
