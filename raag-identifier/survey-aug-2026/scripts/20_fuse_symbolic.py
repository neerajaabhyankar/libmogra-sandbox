"""Stage 5 [hybrid] — fuse a CQT run's logits with motif-classifier's symbolic scores.

The two families cannot be failing the same way. The CQT net reads a Sa-anchored
spectrogram and never sees a pitch track; M14 reads CREPE/Tony note events and never sees
spectral energy. If their errors are even partly independent, a weighted sum of their
per-raag scores beats both, at the cost of one afternoon and no training.

    poetry run python scripts/20_fuse_symbolic.py                     # c4h + M14
    poetry run python scripts/20_fuse_symbolic.py --dl aug_jitter     # a different DL run
    poetry run python scripts/20_fuse_symbolic.py --symbolic m12      # a different partner

**Protocol.** The symbolic method is fitted on exactly the clips the DL model was fitted on
(the 1350-clip fit half of the same video-grouped split, same seed), then scores the same
460 val clips and the same 150 test clips. Both score matrices are turned into probabilities
with a softmax at a temperature fitted on val, so the two are on a common footing before
they are added — motif-classifier's scores are unnormalised affinities and the DL logits are
not calibrated either, and summing raw scores would just let whichever has the larger scale
win.

**The mixing weight is swept on val and applied once to test.** The sweep is a choice made
on validation data, exactly like a hyperparameter; test sees one number.

Nothing here retrains anything. It reads `best.pt` for the DL side and the note cache for
the symbolic side.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from common import metrics, trainer
from common.data import grouped_split, labels, load_clips
from common.paths import MOTIF_DIR, RESULTS, add_sibling_paths

_train = None          # imported lazily; pulls torch models in


@contextlib.contextmanager
def _shadowed(*names):
    """Run a block with `names` evicted from `sys.modules`, then put them back.

    Both projects define a top-level package called `models`: ours under
    `survey-aug-2026/models/`, and melody-extraction's under
    `melody-first/sequence/models/` (which `represent.py` reaches through
    `pipeline.estimate_tonic_hz`). Whichever is imported first wins for the whole process,
    so importing our trainer and then motif-classifier gives
    `ModuleNotFoundError: models.gamadhani` -- our package, correctly, has no such module.

    Rather than rename anyone's package, isolate: evict the cached `models` while the
    symbolic side imports, then restore ours. Both halves get the package they mean.
    """
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k in names or any(k.startswith(n + ".") for n in names)}
    try:
        yield
    finally:
        for k in [k for k in sys.modules
                  if k in names or any(k.startswith(n + ".") for n in names)]:
            del sys.modules[k]
        sys.modules.update(saved)


def _dl_scores(run_id, clips_by_split, device="auto"):
    """{split: (n_clips, 50) logits} from a finished run's checkpoint. No retraining."""
    global _train
    if _train is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from importlib import import_module
        _train = import_module("10_train")

    run_dir = RESULTS / run_id
    result = json.loads((run_dir / "result.json").read_text())

    class _Args:
        def __init__(self, cfg):
            self.__dict__.update(cfg)

    args = _Args(result["config"])
    args.device = device
    cfg = trainer.TrainConfig(batch_size=getattr(args, "batch_size", 16), num_workers=0,
                              device=device, seed=getattr(args, "seed", 0))
    model = _train.build_model(args)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=cfg.device,
                                     weights_only=True))
    model.to(cfg.device)

    out = {}
    for split, clips in clips_by_split.items():
        ds = _train.make_dataset(args.arch, clips, args, None, train=False)
        out[split] = trainer.predict(model, ds, cfg)
    return out, result


def _melody_scores(fit_clips, clips_by_split, label_order):
    """`--symbolic melody`: the naive 120-bin histogram + logreg, fitted on the same clips.

    The portable partner. M14 needs both pitch trackers -- pYIN through the `vamp` native
    plugin for its notes, and torchcrepe for its histogram -- which is a heavy dependency
    for something meant to be handed to someone else. This branch needs only torchcrepe,
    and the survey already measured the two as equally strong on test (0.373 each), so it
    is worth knowing whether it fuses as well as M14 does.
    """
    from importlib import import_module

    probe = import_module("01_probe_representations")   # its logreg, not a second copy
    from common import melody

    y = np.array([label_order.index(c.raag) for c in fit_clips])
    X = melody.cached(fit_clips)
    return {split: probe._logreg_scores(X, y, melody.cached(clips), len(label_order))
            for split, clips in clips_by_split.items()}


def _symbolic_scores(method_name, fit_clips, clips_by_split, label_order):
    """{split: (n_clips, 50) scores} from motif-classifier, fitted on `fit_clips`.

    Fitted on the DL model's own fit half so the fusion weight swept on val is honest --
    a symbolic model that had seen the val clips would make val look better than test.
    """
    add_sibling_paths()
    with _shadowed("models"):
        from evaluate import get_feats, make_method        # noqa: E402
        from represent import Params                        # noqa: E402
        return _symbolic_inner(get_feats, make_method, Params, method_name,
                               fit_clips, clips_by_split, label_order)


def _symbolic_inner(get_feats, make_method, Params, method_name, fit_clips,
                    clips_by_split, label_order):

    cfg = json.loads((MOTIF_DIR / "results" / "v1.1" / "final.json").read_text())
    if method_name not in cfg:
        raise KeyError(f"{method_name} not in motif-classifier final.json; "
                       f"have {sorted(cfg)}")
    # final.json is {method: {config: {rep, method, features, ...}, test: ..., ...}}
    spec = cfg[method_name]["config"]
    rep = Params(**spec["rep"])
    builder = {"m14": "m9plus"}.get(method_name, method_name)
    feature_kw = spec.get("features") or None

    feats_train = get_feats(rep, split="train", feature_kw=feature_kw)
    feats_test = get_feats(rep, split="test", feature_kw=feature_kw)
    by_id = {f.clip.clip_id: f for f in list(feats_train) + list(feats_test)}

    fit_ids = {c.clip_id for c in fit_clips}
    method = make_method(builder, **spec["method"])
    method.fit([f for f in feats_train if f.clip.clip_id in fit_ids])

    idx = {r: i for i, r in enumerate(method.raags)}
    order = [idx[r] for r in label_order]        # motif's raag order -> ours
    out = {}
    for split, clips in clips_by_split.items():
        S = np.full((len(clips), len(label_order)), np.nan, dtype=np.float32)
        for i, c in enumerate(clips):
            f = by_id.get(c.clip_id)
            if f is None:
                continue
            s, _k = method.score(f)
            S[i] = np.asarray(s, dtype=np.float32)[order]
        missing = int(np.isnan(S[:, 0]).sum())
        if missing:
            print(f"  {split}: {missing}/{len(clips)} clips had no symbolic features; "
                  f"filled with the row mean (they fall back to the DL model)")
            S[np.isnan(S[:, 0])] = np.nanmean(S, axis=0)
        out[split] = S
    return out


def _probs(scores, temperature):
    z = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-6)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_temperature(scores, clips, label_order, grid=None):
    """Temperature minimising NLL, so two differently-scaled score matrices can be added."""
    y = np.array([label_order.index(c.raag) for c in clips])
    best, best_nll = 1.0, np.inf
    for T in (grid if grid is not None else np.geomspace(0.01, 100.0, 60)):
        p = _probs(scores, T)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best_nll:
            best, best_nll = float(T), float(nll)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dl", default="c4h", help="a finished run under results/v1.1/")
    ap.add_argument("--symbolic", default="m14",
                   help="a method in motif-classifier's final.json, or 'melody' for the "
                        "naive histogram + logreg (no motif-classifier, no vamp plugin)")
    ap.add_argument("--run-id", default=None, help="where to write; default fuse_<dl>_<sym>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    run_id = a.run_id or f"fuse_{a.dl}_{a.symbolic}"
    out_dir = RESULTS / run_id
    L = labels()
    train_pool = load_clips("train")
    seed = json.loads((RESULTS / a.dl / "result.json").read_text())["config"].get("seed", 0)
    fit_clips, val_clips = grouped_split(train_pool, val_frac=0.2, seed=seed)
    test_clips = load_clips("test")
    splits = {"val": val_clips, "test": test_clips}

    print(f"=== {run_id} ===")
    print(f"  DL       : {a.dl}  (seed {seed}, fitted on {len(fit_clips)} clips)")
    print(f"  symbolic : {a.symbolic}  (refitted on the same {len(fit_clips)} clips)")
    print(f"  scoring  : {len(val_clips)} val, {len(test_clips)} test")
    if a.dry_run:
        print(f"  would write -> {out_dir}")
        return

    dl, dl_result = _dl_scores(a.dl, splits, device=a.device)
    sym = (_melody_scores(fit_clips, splits, L) if a.symbolic == "melody"
           else _symbolic_scores(a.symbolic, fit_clips, splits, L))

    T_dl = _fit_temperature(dl["val"], val_clips, L)
    T_sym = _fit_temperature(sym["val"], val_clips, L)
    print(f"  calibration: T_dl {T_dl:.3f}  T_symbolic {T_sym:.3f}")

    P = {s: {"dl": _probs(dl[s], T_dl), "sym": _probs(sym[s], T_sym)} for s in splits}

    print(f"\n  {'w':>5}  {'val top-1':>9}   (w = weight on the symbolic model)")
    best_w, best_val, sweep = 0.0, -1.0, []
    for w in np.round(np.arange(0.0, 1.01, 0.05), 2):
        m, _rows = metrics.score(val_clips, np.log(np.clip(
            (1 - w) * P["val"]["dl"] + w * P["val"]["sym"], 1e-12, None)))
        sweep.append((float(w), m["top1"]))
        flag = ""
        if m["top1"] > best_val:
            best_w, best_val, flag = float(w), m["top1"], "  <-"
        if abs(w * 20 - round(w * 20)) < 1e-9 and int(round(w * 20)) % 2 == 0:
            print(f"  {w:>5.2f}  {m['top1']:>9.3f}{flag}")

    print(f"\n  chosen on val: w = {best_w:.2f}  (val top-1 {best_val:.3f})")
    print(f"    w=0 is the DL model alone : {sweep[0][1]:.3f}")
    print(f"    w=1 is the symbolic alone : {sweep[-1][1]:.3f}")

    fused = {s: np.log(np.clip((1 - best_w) * P[s]["dl"] + best_w * P[s]["sym"], 1e-12, None))
             for s in splits}
    val_m, val_rows = metrics.score(val_clips, fused["val"])
    T = metrics.calibrate_temperature(val_rows)
    te_m, te_rows = metrics.score(test_clips, fused["test"])

    print(f"\n  val  {metrics.summary_line(val_m)}")
    print(f"  TEST {metrics.summary_line(te_m)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": run_id, "stage": 5, "arch": "fusion",
        "data_revision": dl_result["data_revision"],
        "config": {"dl_run": a.dl, "symbolic": a.symbolic, "weight": best_w,
                   "T_dl": T_dl, "T_symbolic": T_sym, "seed": seed,
                   "sweep": sweep, "arch": "fusion"},
        "split": "grouped-val", "metrics": val_m,
        "musical": metrics.musical(val_rows, temperature=T), "temperature": T,
        "test": {"metrics": te_m, "musical": metrics.musical(te_rows, temperature=T)},
        "best_epoch": None, "wall_clock_s": 0,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    np.save(out_dir / "test_logits.npy", fused["test"])
    metrics.confusion(te_rows, out_dir / "confusion_test.png",
                      title=f"{run_id} test — top-1 {te_m['top1']:.3f}")
    print(f"\nwrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
