"""Train one configuration of one architecture. Every Stage 1-4 experiment is a call to this.

There is one training script rather than one per experiment on purpose: the stages differ
only in flags, and a per-stage script family is how two runs end up accidentally using
different splits, different normalisation or different metrics, at which point the
comparison between them means nothing.

    # Stage 1 -- retrain as-is on v1.1
    python scripts/10_train.py --arch hubert   --run-id d1
    python scripts/10_train.py --arch resnet1d --run-id r1
    python scripts/10_train.py --arch cqt      --run-id c1

    # Stage 2 -- the tonic, three ways, plus the control that catches plumbing bugs
    python scripts/10_train.py --arch hubert --run-id d2n --tonic normalise
    python scripts/10_train.py --arch hubert --run-id d2c --tonic-mode condition
    python scripts/10_train.py --arch cqt    --run-id c2  --tonic normalise
    python scripts/10_train.py --arch cqt    --run-id c2_shuffled --tonic normalise --shuffle-tonics

    # Stage 3 -- source separation in front
    python scripts/10_train.py --arch cqt --run-id c3_hpss --tonic normalise --separate hpss

    # Stage 4 -- the libmogra DB as a prior
    python scripts/10_train.py --arch cqt --run-id c4_graded --tonic normalise --graded-alpha 0.3
    python scripts/10_train.py --arch cqt --run-id c4_aux    --tonic normalise --aux-weight 0.2

    # Stage 5 -- the hybrid: the naive melody histogram alongside the learned feature
    python scripts/10_train.py --arch cqt --run-id hybrid_feat --tonic normalise --melody

    # the CQT trunk's shape and how it summarises time (Batch 9)
    python scripts/10_train.py --arch cqt --run-id pool_tconv --tonic normalise --db-head --cqt-pool tconv
    python scripts/10_train.py --arch cqt --run-id arch_w2    --tonic normalise --db-head --cqt-width 2

    # train on the full recordings instead of the Hub clips; val/test stay the Hub clips (Batch 10)
    python scripts/10_train.py --arch cqt --run-id full_aug --tonic normalise --db-head \
        --train-source full --windows-per-video 20

    # 5-fold grouped CV instead of a single val split (affordable for cqt, not for hubert)
    python scripts/10_train.py --arch cqt --run-id c2_cv --tonic normalise --folds 5

    # score the held-out 150 -- explicit, logged in RUNS.md, once per method
    python scripts/10_train.py --arch cqt --run-id c2 --tonic normalise --test

Resumable: re-running the same --run-id picks up from `state.pt`. Add --fresh to start over.
Nothing is ever pushed to the Hub.
"""

import argparse
import json
import sys
import time
from functools import lru_cache

import numpy as np
import torch

import _bootstrap  # noqa: F401
import models
from common import audio, datasets, melody, metrics, trainer
from common.data import fold_indices, grouped_split, labels, load_clips, summarise
from common.losses import Objective
from common.paths import DATA_REVISION, RESULTS
from common.tonic import shuffled_tonics

#: what each architecture wants its audio to look like
ARCH_INPUT = {
    "hubert":   dict(kind="wave", sr=audio.SR_HUBERT, channels=1),
    "resnet1d": dict(kind="wave", sr=audio.SR_JEEVSTER, channels=2),
    "cqt":      dict(kind="cqt"),
}


@lru_cache(maxsize=1)
def melody_side():
    """{clip_id: (120,)} naive melody histograms for every clip, built once per process.

    Read from disk rather than passed in, so that the two scripts that rebuild a finished
    model from its recorded config -- `91_score_test.py` and `20_fuse_symbolic.py` -- get
    the side vector without knowing it exists.
    """
    return melody.by_clip_id(load_clips())


def full_audio_train_set(clips, args, tonic_override):
    """The *same* fit videos as `clips`, read from their full recordings instead of their
    five Hub clips each. Validation and test are never drawn from here -- they stay the Hub
    clips, so a full-audio run is selected and scored exactly like every other run."""
    from common import fullaudio
    from common.paths import SPLITS_FILE

    if args.arch != "cqt":
        raise ValueError("--train-source full is implemented for the CQT architecture only")
    if tonic_override or args.separate or getattr(args, "melody", False):
        raise ValueError("--train-source full supports neither the shuffled-tonic control, "
                         "--separate, nor --melody: the cache is one anchored CQT per video")
    index = fullaudio.by_id()
    vids = sorted({c.video for c in clips})
    wrong = [v for v in vids if index[v].role(args.seed) != "fit"]
    if wrong:
        raise RuntimeError(f"{len(wrong)} training videos are not 'fit' under seed "
                           f"{args.seed} in {SPLITS_FILE.name}: the two split derivations "
                           f"disagree, and nothing should train until that is understood")
    return fullaudio.FullAudioCQTDataset(
        [index[v] for v in vids], args.windows_per_video, train=True,
        freq_shift_bins=args.freq_jitter, trim_seconds=args.trim_seconds,
        loud_fraction=args.loud_fraction, loudness_db=args.loudness_db,
        fixed_pool=getattr(args, "window_pool", False))


def make_dataset(arch, clips, args, tonic_override, train):
    if train and getattr(args, "train_source", "hub") == "full":
        return full_audio_train_set(clips, args, tonic_override)
    spec = ARCH_INPUT[arch]
    common_kw = dict(tonic=args.tonic, separate=args.separate, seconds=args.seconds,
                     length_policy=args.length_policy, tonic_override=tonic_override,
                     train=train, side=melody_side() if getattr(args, "melody", False) else None)
    if spec["kind"] == "wave":
        return datasets.WaveformDataset(clips, spec["sr"], channels=spec["channels"],
                                        gain_jitter_db=args.gain_jitter if train else 0.0,
                                        **common_kw)
    return datasets.CQTDataset(clips, time_frames=args.cqt_frames,
                               freq_shift_bins=args.freq_jitter if train else 0,
                               **common_kw)


def build_model(args):
    """The default classifier head, or -- with --db-head -- one that scores against the
    libmogra templates (M12's mechanism, learned end to end).

    `--melody` widens whichever head is in use by an encoded melody histogram."""
    arch_kw = {}
    melody_on = getattr(args, "melody", False)      # absent from pre-Stage-5 configs
    side_kw = dict(side_dim=len(next(iter(melody_side().values()))) if melody_on else 0,
                   side_out=getattr(args, "melody_dim", 64))
    if args.arch == "hubert":
        arch_kw.update(freeze_encoder=not args.unfreeze_encoder)
    elif args.arch == "resnet1d":
        arch_kw.update(unfreeze_blocks=args.unfreeze_blocks)
    elif args.arch == "cqt":
        arch_kw.update(fold_octaves=args.fold_octaves,
                       pool=getattr(args, "cqt_pool", "mean"),
                       width=getattr(args, "cqt_width", 1.0),
                       depth=getattr(args, "cqt_depth", 4),
                       freq_pools=getattr(args, "cqt_freq_pools", 3))

    if args.db_head:
        from models.dbhead import RaagClassifierDB

        backbone = models.build_backbone(args.arch, **arch_kw)
        return RaagClassifierDB(backbone, backbone.out_dim, num_labels=len(labels()),
                                tonic_mode=args.tonic_mode, lam=args.db_lam,
                                n_bins=args.db_bins,
                                learn_templates=not args.db_freeze_templates, **side_kw)
    return models.build(args.arch, num_labels=len(labels()), tonic_mode=args.tonic_mode,
                        aux_occupancy=args.aux_weight > 0, **arch_kw, **side_kw)


def fit_once(args, train_clips, val_clips, out_dir, tonic_override, cfg):
    model = build_model(args)
    tr_ds = make_dataset(args.arch, train_clips, args, tonic_override, train=True)
    va_ds = make_dataset(args.arch, val_clips, args, tonic_override, train=False)
    loss_fn = Objective(graded_alpha=args.graded_alpha, graded_gamma=args.graded_gamma,
                        aux_weight=args.aux_weight, device=cfg.device)
    groups = models.param_groups(args.arch, model, lr=args.lr, head_lr=args.head_lr,
                                 weight_decay=args.weight_decay)
    history, best = trainer.train(model, tr_ds, va_ds, cfg, out_dir, loss_fn=loss_fn,
                                  param_groups=groups, resume=not args.fresh)
    return model, history, best, loss_fn


class _Tee:
    """stdout duplicated to a file, line-buffered."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        self._stream.write(s)
        self._fh.write(s)
        self._fh.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _tee_stdout_to(path):
    """Every run writes its own log into its own directory, however it was launched.

    The batch scripts pipe through `tee`, but a run started by hand
    (`python scripts/10_train.py --run-id x ...`) used to leave nothing behind except
    wherever the shell happened to redirect it -- so `scripts/status.sh` could not find it
    and neither could you, a week later. Appends, so a resumed run keeps its history.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=list(ARCH_INPUT))
    ap.add_argument("--run-id", required=True, help="directory under results/v1.1/")
    ap.add_argument("--stage", type=int, default=0, help="for the results table only")

    g = ap.add_argument_group("the tonic")
    g.add_argument("--tonic", default="none", choices=["none", "normalise"],
                   help="audio-level: resample so Sa lands on a reference (cqt: use the "
                        "Sa-anchored CQT instead, which needs no resampling)")
    g.add_argument("--tonic-mode", default="none", choices=["none", "condition"],
                   help="model-level: FiLM the pooled feature by the tonic vector")
    g.add_argument("--shuffle-tonics", action="store_true",
                   help="CONTROL: permute tonics between videos. A tonic experiment whose "
                        "score does not drop under this is not using the tonic")

    g = ap.add_argument_group("input")
    g.add_argument("--separate", default=None, help="hpss | hpss+drone | demucs")
    g.add_argument("--seconds", type=float, default=audio.DEFAULT_SECONDS)
    g.add_argument("--length-policy", default="fixed", choices=["fixed", "musical"])
    g.add_argument("--cqt-frames", type=int, default=431)
    g.add_argument("--gain-jitter", type=float, default=0.0, help="dB, train only")
    g.add_argument("--melody", action="store_true",
                   help="STAGE 5 HYBRID: concatenate the naive melody histogram "
                        "(common/melody.py -- 120 bins of CREPE pitch mass against the "
                        "annotated Sa) onto the pooled feature, and train the two "
                        "together. Works with any arch and either head")
    g.add_argument("--melody-dim", type=int, default=64,
                   help="width the melody histogram is encoded to before concatenation")
    g.add_argument("--freq-jitter", type=int, default=0,
                   help="CQT bins of random pitch jitter, train only; keep under a semitone "
                        "(36 bins/octave -> 3 bins per semitone)")

    g = ap.add_argument_group("the database as a prior")
    g.add_argument("--graded-alpha", type=float, default=0.0,
                   help="mass moved from the one-hot onto musically adjacent raags")
    g.add_argument("--graded-gamma", type=float, default=4.0)
    g.add_argument("--aux-weight", type=float, default=0.0,
                   help="weight on the auxiliary swar-occupancy head")
    g.add_argument("--db-head", action="store_true",
                   help="replace the Linear(D,50) head with one that predicts a swar "
                        "profile and scores it against the DB templates by chi-square -- "
                        "M12's mechanism, learned end to end")
    g.add_argument("--db-lam", type=float, default=0.3,
                   help="how far the templates are pulled toward the database. 0 = learned "
                        "only, 1 = the database verbatim. M12's optimum was 0.3")
    g.add_argument("--db-bins", type=int, default=12, choices=[12, 36, 144])
    g.add_argument("--db-freeze-templates", action="store_true",
                   help="with --db-lam 1, leaves 50 scalar biases as the only raag-specific "
                        "parameters in the model")

    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--patience", type=int, default=8)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--lr", type=float, default=None, help="backbone lr; per-arch default")
    g.add_argument("--head-lr", type=float, default=None)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--select-on", default="top1",
                   choices=["val_loss", "top1", "top5", "macro_f1"],
                   help="which val metric picks the checkpoint. NOT val_loss by default -- "
                        "see the note in common/trainer.py")
    g.add_argument("--unfreeze-blocks", type=int, default=2, help="resnet1d only")
    g.add_argument("--unfreeze-encoder", action="store_true", help="hubert only; needs a GPU")
    g.add_argument("--fold-octaves", action="store_true", help="cqt only")
    g.add_argument("--cqt-pool", default="mean",
                   choices=["mean", "stats", "attn", "tconv", "gru"],
                   help="cqt only: how the trunk's ~26 positions are summarised over time. "
                        "tconv/gru are order-sensitive; stats/attn are their controls. See "
                        "models/cqtnet.py")
    g.add_argument("--cqt-width", type=float, default=1.0,
                   help="cqt only: channel multiplier on (32, 64, 96, 128) and the 24-ch proj")
    g.add_argument("--cqt-depth", type=int, default=4, help="cqt only: residual blocks")
    g.add_argument("--cqt-freq-pools", type=int, default=3,
                   help="cqt only: blocks that halve frequency. 3 -> 18 output cells over 4 "
                        "octaves (~2.7 semitones each), 2 -> 36, 1 -> 72")

    g = ap.add_argument_group("training data")
    g.add_argument("--train-source", default="hub", choices=["hub", "full"],
                   help="hub = the dataset's clips. full = random 20 s windows of the same "
                        "fit videos' full recordings (common/fullaudio.py). val/test are "
                        "the Hub clips either way")
    g.add_argument("--windows-per-video", type=int, default=20,
                   help="full only: windows drawn per video per epoch; the Hub has 5")
    g.add_argument("--window-pool", action="store_true",
                   help="full only: train on a FIXED set of --windows-per-video evenly "
                        "spaced windows per video, reused every epoch -- a dataset with N "
                        "clips per performance -- instead of fresh random windows")
    g.add_argument("--trim-seconds", type=float, default=30.0,
                   help="full only: never start a window this close to either end")
    g.add_argument("--loud-fraction", type=float, default=0.8,
                   help="full only: share of a window's frames that must be loud. 0 = off")
    g.add_argument("--loudness-db", type=float, default=20.0,
                   help="full only: 'loud' = within this many dB of the median frame")

    g = ap.add_argument_group("protocol")
    g.add_argument("--folds", type=int, default=1,
                   help="1 = a single grouped train/val split; N>1 = grouped N-fold CV")
    g.add_argument("--no-test", dest="test", action="store_false", default=True,
                   help="skip scoring the held-out 150 at the end of the run")
    g.add_argument("--test", dest="test", action="store_true",
                   help=argparse.SUPPRESS)   # kept so old commands still parse
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto")
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--fresh", action="store_true", help="ignore any checkpoint and restart")
    g.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.lr is None:
        args.lr = {"hubert": 1e-4, "resnet1d": 1e-4, "cqt": 1e-3}[args.arch]

    out_dir = RESULTS / args.run_id
    if not args.dry_run:
        _tee_stdout_to(out_dir / "run.log")
    train_clips = load_clips("train")
    tonic_override = shuffled_tonics(load_clips(), seed=args.seed) if args.shuffle_tonics else None

    cfg = trainer.TrainConfig(epochs=args.epochs, patience=args.patience,
                              batch_size=args.batch_size, lr=args.lr,
                              weight_decay=args.weight_decay, select_on=args.select_on,
                              num_workers=args.num_workers, seed=args.seed,
                              device=args.device)

    print(f"=== {args.run_id} | arch={args.arch} | {DATA_REVISION[:10]} ===")
    print(f"  train pool: {summarise(train_clips)}")
    print(f"  tonic: audio={args.tonic} model={args.tonic_mode}"
          f"{' SHUFFLED-CONTROL' if args.shuffle_tonics else ''} | separate={args.separate} "
          f"| length={args.length_policy}")
    if args.train_source == "full":
        kind = "fixed pool of" if args.window_pool else "fresh"
        print(f"  training data: FULL RECORDINGS, {kind} {args.windows_per_video} windows/video/epoch, "
              f"trim {args.trim_seconds:.0f}s, loud fraction {args.loud_fraction}")
    if args.arch == "cqt" and (args.cqt_pool, args.cqt_width, args.cqt_depth,
                               args.cqt_freq_pools) != ("mean", 1.0, 4, 3):
        print(f"  cqt trunk: pool={args.cqt_pool} width={args.cqt_width} "
              f"depth={args.cqt_depth} freq_pools={args.cqt_freq_pools}")
    if args.melody:
        print(f"  hybrid: melody histogram -> {args.melody_dim} dims, concatenated onto "
              f"the pooled feature")
    print(f"  db prior: graded_alpha={args.graded_alpha} aux_weight={args.aux_weight}"
          + (f" | DB-template head lam={args.db_lam} bins={args.db_bins}"
             f"{' frozen' if args.db_freeze_templates else ''}" if args.db_head else ""))
    print(f"  protocol: {'%d-fold grouped CV' % args.folds if args.folds > 1 else 'single grouped val split'}"
          f"{' + TEST' if args.test else ''}")
    if args.dry_run:
        print(f"  would write -> {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_classes = len(labels())

    if args.folds > 1 and args.train_source == "full":
        raise ValueError("--train-source full is not wired for --folds; use seeds instead")
    if args.folds > 1:
        oof = np.zeros((len(train_clips), n_classes), dtype=np.float32)
        index = {c.clip_id: i for i, c in enumerate(train_clips)}
        histories, model = [], None
        for k, tr, va in fold_indices(train_clips, n_folds=args.folds, seed=args.seed):
            print(f"\n-- fold {k}: train {summarise(tr)} | val {summarise(va)}")
            model, hist, best, loss_fn = fit_once(args, tr, va, out_dir / f"fold{k}",
                                                  tonic_override, cfg)
            va_ds = make_dataset(args.arch, va, args, tonic_override, train=False)
            oof[[index[c.clip_id] for c in va]] = trainer.predict(model, va_ds, cfg)
            histories.append({"fold": k, "best": best, "history": hist})
        val_m, val_rows = metrics.score(train_clips, oof)
        eval_clips, history, best = train_clips, histories, {"epoch": None}
        split_name = f"grouped-{args.folds}fold-cv"
    else:
        tr, va = grouped_split(train_clips, val_frac=0.2, seed=args.seed)
        print(f"  fit on {summarise(tr)} | select on {summarise(va)}")
        model, history, best, loss_fn = fit_once(args, tr, va, out_dir, tonic_override, cfg)
        va_ds = make_dataset(args.arch, va, args, tonic_override, train=False)
        val_m, val_rows = metrics.score(va, trainer.predict(model, va_ds, cfg))
        eval_clips = va
        split_name = "grouped-val"

    T = metrics.calibrate_temperature(val_rows)
    val_mus = metrics.musical(val_rows, temperature=T)
    print(f"\n{split_name}: {metrics.summary_line(val_m)}")
    print(f"  graded: mistake_affinity {val_mus['mistake_affinity']:.3f} "
          f"(chance {val_mus['mistake_affinity_chance']:.3f}) | "
          f"tonic_explained {val_mus['tonic_explained']:.3f} "
          f"(chance {val_mus['tonic_explained_chance']:.3f})")

    result = {
        "run_id": args.run_id, "stage": args.stage, "arch": args.arch,
        "data_revision": DATA_REVISION, "config": vars(args),
        "split": split_name, "metrics": val_m, "musical": val_mus, "temperature": T,
        "best_epoch": best.get("epoch"), "wall_clock_s": round(time.time() - t0, 1),
    }

    if args.test:
        test_clips = load_clips("test")
        te_ds = make_dataset(args.arch, test_clips, args, tonic_override, train=False)
        te_logits = trainer.predict(model, te_ds, cfg)
        te_m, te_rows = metrics.score(test_clips, te_logits)
        te_mus = metrics.musical(te_rows, temperature=T)
        print(f"\nTEST (scored once): {metrics.summary_line(te_m)}")
        result["test"] = {"metrics": te_m, "musical": te_mus}
        metrics.confusion(te_rows, out_dir / "confusion_test.png",
                          title=f"{args.run_id} test — top-1 {te_m['top1']:.3f}")
        np.save(out_dir / "test_logits.npy", te_logits)

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    metrics.confusion(val_rows, out_dir / f"confusion_{split_name}.png",
                      title=f"{args.run_id} {split_name} — top-1 {val_m['top1']:.3f}")
    if args.folds == 1:
        trainer.plot_curves(history, best["epoch"], args.run_id, out_dir / "curves.png")
    print(f"\nwrote {out_dir / 'result.json'} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
