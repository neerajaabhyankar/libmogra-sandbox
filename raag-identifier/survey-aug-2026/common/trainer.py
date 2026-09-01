"""One training loop for all three architectures.

Deliberately small and boring. What it does provide, because every run in this project needs
it and getting any of them wrong invalidates the comparison:

  * **video-grouped evaluation** every epoch, through `common.metrics`, so the number in the
    log is the same number that ends up in the results table
  * **resume from checkpoint** -- `state.pt` is rewritten every epoch, so a 4-hour job that
    dies at hour 3, or a Colab runtime that evaporates, restarts where it stopped
  * **early stopping on a chosen metric**, with the best epoch's weights kept separately
    from the last epoch's
  * `result.json` in the schema RUNS.md documents, so runs are comparable without reading
    the script that produced them

    hist = train(model, train_ds, val_ds, TrainConfig(...), out_dir)
"""

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import metrics
from .datasets import loader


def pick_device(prefer=None):
    if prefer and prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainConfig:
    epochs: int = 30
    patience: int = 8
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    select_on: str = "top1"          # "val_loss" | "top1" | "top5" | "macro_f1"
    #  Why top1 and not val_loss: on this corpus a 50-class model's val cross-entropy
    #  bottoms out within a couple of epochs and then climbs as the model grows confident,
    #  while val top-1 keeps improving for another five or ten. Selecting on loss picks a
    #  checkpoint that is barely above chance (measured: c1 epoch 1, top-1 0.017, against
    #  0.070 at epoch 6). Selecting on the reported metric does bias the val column upward
    #  a little -- the held-out test split is the unbiased number, and it is scored once.
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    log_every: int = 50              # steps

    def __post_init__(self):
        self.device = pick_device(self.device)


def _forward(model, batch, device):
    """Models here return a dict with at least 'logits'. Anything extra (an auxiliary
    occupancy head) is passed through to the objective untouched."""
    out = model(input_values=batch["input_values"].to(device), tonic=batch["tonic"].to(device))
    return out if isinstance(out, dict) else {"logits": out.logits}


@torch.no_grad()
def predict(model, dataset, cfg, batch_size=None):
    """(n_clips, n_classes) logits, in dataset order."""
    model.eval()
    dl = loader(dataset, batch_size or cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers)
    out = np.zeros((len(dataset), 50), dtype=np.float32)
    for batch in dl:
        logits = _forward(model, batch, cfg.device)["logits"]
        out[batch["index"].numpy()] = logits.float().cpu().numpy()
    return out


def evaluate(model, dataset, cfg, loss_fn=None):
    """Metrics on a dataset, plus mean loss if an objective is given."""
    logits = predict(model, dataset, cfg)
    m, rows = metrics.score(dataset.clips, logits)
    if loss_fn is not None:
        y = torch.tensor([c.label for c in dataset.clips])
        with torch.no_grad():
            loss, _ = loss_fn({"logits": torch.from_numpy(logits)}, {"labels": y})
        m["loss"] = float(loss)
    return m, rows


def train(model, train_ds, val_ds, cfg: TrainConfig, out_dir, loss_fn=None,
          param_groups=None, resume=True, on_epoch=None):
    """Fit, with early stopping and resume. Returns the history dict."""
    from .losses import Objective

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    loss_fn = loss_fn or Objective(device=cfg.device)
    model.to(cfg.device)

    groups = param_groups or [{"params": [p for p in model.parameters() if p.requires_grad],
                               "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    opt = torch.optim.AdamW(groups)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  device={cfg.device}  trainable {n_train:,}/{n_all:,} "
          f"({100 * n_train / max(n_all, 1):.0f}%)  objective: {loss_fn.describe()}")

    history = {k: [] for k in ("train_loss", "val_loss", "val_top1", "val_top5",
                               "val_macro_f1", "val_mrr", "epoch_seconds")}
    start_epoch, best = 0, {"score": None, "epoch": -1}
    state_path = out_dir / "state.pt"
    if resume and state_path.exists():
        st = torch.load(state_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        history, start_epoch, best = st["history"], st["epoch"] + 1, st["best"]
        print(f"  resuming from epoch {start_epoch} (best {best['score']} @ {best['epoch']})")

    dl = loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    better = (lambda a, b: a < b) if cfg.select_on == "val_loss" else (lambda a, b: a > b)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(dl):
            opt.zero_grad()
            loss, _parts = loss_fn(_forward(model, batch, cfg.device), batch)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], cfg.grad_clip)
            opt.step()
            bs = batch["labels"].shape[0]
            total += float(loss.detach()) * bs
            n += bs
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"    e{epoch} step {step}/{len(dl)} loss {float(loss.detach()):.3f} "
                      f"({(time.time() - t0) / max(step, 1):.2f}s/step)", flush=True)

        vm, _rows = evaluate(model, val_ds, cfg, loss_fn)
        history["train_loss"].append(total / max(n, 1))
        history["val_loss"].append(vm["loss"])
        for k in ("top1", "top5", "macro_f1", "mrr"):
            history[f"val_{k}"].append(vm[k])
        history["epoch_seconds"].append(time.time() - t0)
        print(f"  epoch {epoch}: train_loss {total / max(n, 1):.3f} | "
              f"val_loss {vm['loss']:.3f} | {metrics.summary_line(vm)} "
              f"[{time.time() - t0:.0f}s]", flush=True)

        score = vm["loss"] if cfg.select_on == "val_loss" else vm[cfg.select_on]
        if best["score"] is None or better(score, best["score"]):
            best = {"score": float(score), "epoch": epoch}
            torch.save(model.state_dict(), out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "history": history, "epoch": epoch, "best": best}, state_path)
        (out_dir / "history.json").write_text(json.dumps(
            {"history": history, "best": best, "config": asdict(cfg)}, indent=2))
        if on_epoch:
            on_epoch(epoch, model, vm)
        if cfg.patience and epoch - best["epoch"] >= cfg.patience:
            print(f"  early stop at epoch {epoch} (best was {best['epoch']})")
            break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=cfg.device, weights_only=True))
    return history, best


def plot_curves(history, best_epoch, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.arange(len(history["train_loss"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(e, history["train_loss"], label="train")
    ax[0].plot(e, history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("loss")
    ax[1].plot(e, history["val_top1"], label="val top-1")
    ax[1].plot(e, history["val_top5"], label="val top-5", linestyle=":")
    ax[1].plot(e, history["val_macro_f1"], label="val macro-F1", linestyle="--")
    ax[1].axhline(0.02, color="grey", lw=0.8, label="chance")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("val metrics")
    for a in ax:
        a.axvline(best_epoch, color="grey", ls="--", lw=0.8)
    fig.suptitle(f"{title} (best epoch {best_epoch})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
