# Stage 1 hyperparameter sweep over the MLP head (see plan.md).
#
# Same data/CV setup as 01_probe_embeddings.py (clip_mean 300-dim embeddings, 123
# train clips, 5 classes, stratified 5-fold CV with the SAME folds across all
# configs for a fair comparison). Sweeps:
#   - depth:        0 (linear probe), 1, 2 hidden layers
#   - width:        64, 128, 256        (depth >= 1 only)
#   - batchnorm:    on / off            (depth >= 1 only)
#   - dropout:      0.0, 0.2, 0.5       (depth >= 1 only)
#   - weight decay: 0.0, 1e-4, 1e-2
#   - lr:           1e-3, 3e-4
#
# For each config:
#   - run CV, record mean/std accuracy & macro-F1, and the best epoch (early
#     stopping point) per fold
#   - train a final model on all 123 train clips for `median(best_epoch)+1`
#     epochs (no val set available for early stopping on the full set)
#   - save a checkpoint (state_dict + scaler + config + CV metrics) under
#     outputs/sweep/checkpoints/
#
# Outputs:
#   - outputs/sweep/results.csv          -- all configs ranked by mean CV macro-F1
#   - outputs/sweep/checkpoints/*.pt      -- one checkpoint per config
#   - outputs/sweep/stage1_best_confusion_matrix.png

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from probe_common import load_split, MLPHead, LABEL_NAMES

OUT_DIR = Path(__file__).parent / "outputs" / "sweep"
CKPT_DIR = OUT_DIR / "checkpoints"

SEED = 0
N_FOLDS = 5
EPOCHS = 200
PATIENCE = 20


def build_configs():
    configs = []
    # Linear probe: no hidden layers, so width/batchnorm/dropout don't apply.
    for wd in (0.0, 1e-4, 1e-2):
        for lr in (1e-3, 3e-4):
            configs.append(dict(depth=0, hidden_dims=(), batchnorm=False, dropout=0.0,
                                 weight_decay=wd, lr=lr))
    # 1- and 2-hidden-layer MLPs.
    for depth in (1, 2):
        for width in (64, 128, 256):
            for batchnorm in (False, True):
                for dropout in (0.0, 0.2, 0.5):
                    for wd in (0.0, 1e-4, 1e-2):
                        for lr in (1e-3, 3e-4):
                            configs.append(dict(
                                depth=depth, hidden_dims=(width,) * depth,
                                batchnorm=batchnorm, dropout=dropout,
                                weight_decay=wd, lr=lr,
                            ))
    return configs


def train_one_fold(X_train, y_train, X_val, y_val, num_classes, cfg):
    scaler = StandardScaler().fit(X_train)
    Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    torch.manual_seed(SEED)
    model = MLPHead(X_train.shape[1], num_classes, hidden_dims=cfg["hidden_dims"],
                     batchnorm=cfg["batchnorm"], dropout=cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(Xv).argmax(dim=1).numpy()

    return val_preds, best_epoch


def train_final(X, y, num_classes, cfg, n_epochs):
    scaler = StandardScaler().fit(X)
    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    torch.manual_seed(SEED)
    model = MLPHead(X.shape[1], num_classes, hidden_dims=cfg["hidden_dims"],
                     batchnorm=cfg["batchnorm"], dropout=cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

    return model, scaler


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_split("train")
    num_classes = len(LABEL_NAMES)
    print(f"Loaded {X.shape[0]} train clips, {X.shape[1]}-dim embeddings, {num_classes} classes")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    configs = build_configs()
    print(f"Sweeping {len(configs)} configs x {N_FOLDS} folds")

    rows = []
    all_val_preds_by_cfg = []

    for cfg_id, cfg in enumerate(configs):
        all_val_preds = np.zeros_like(y)
        fold_accs, fold_f1s, best_epochs = [], [], []

        for train_idx, val_idx in folds:
            val_preds, best_epoch = train_one_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                                                     num_classes, cfg)
            all_val_preds[val_idx] = val_preds
            fold_accs.append(accuracy_score(y[val_idx], val_preds))
            fold_f1s.append(f1_score(y[val_idx], val_preds, average="macro"))
            best_epochs.append(best_epoch)

        cv_acc_mean, cv_acc_std = float(np.mean(fold_accs)), float(np.std(fold_accs))
        cv_f1_mean, cv_f1_std = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
        final_epochs = max(1, int(round(np.median(best_epochs))) + 1)

        final_model, scaler = train_final(X, y, num_classes, cfg, final_epochs)

        ckpt_name = f"cfg_{cfg_id:03d}.pt"
        torch.save({
            "config": cfg,
            "state_dict": final_model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "final_epochs": final_epochs,
            "cv_acc_mean": cv_acc_mean,
            "cv_f1_mean": cv_f1_mean,
        }, CKPT_DIR / ckpt_name)

        rows.append(dict(
            cfg_id=cfg_id, depth=cfg["depth"],
            width=cfg["hidden_dims"][0] if cfg["hidden_dims"] else "",
            batchnorm=cfg["batchnorm"], dropout=cfg["dropout"],
            weight_decay=cfg["weight_decay"], lr=cfg["lr"],
            cv_acc_mean=cv_acc_mean, cv_acc_std=cv_acc_std,
            cv_f1_mean=cv_f1_mean, cv_f1_std=cv_f1_std,
            final_epochs=final_epochs, checkpoint=ckpt_name,
        ))
        all_val_preds_by_cfg.append(all_val_preds)

        if (cfg_id + 1) % 20 == 0 or cfg_id == len(configs) - 1:
            print(f"  {cfg_id + 1}/{len(configs)} configs done")

    rows.sort(key=lambda r: r["cv_f1_mean"], reverse=True)

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved sweep results to {csv_path}")

    print("\nTop 10 configs by mean CV macro-F1:")
    for r in rows[:10]:
        print(f"  cfg_{r['cfg_id']:03d}: depth={r['depth']} width={r['width']} "
              f"bn={r['batchnorm']} dropout={r['dropout']} wd={r['weight_decay']} lr={r['lr']} "
              f"-> acc={r['cv_acc_mean']:.3f}+/-{r['cv_acc_std']:.3f} "
              f"f1={r['cv_f1_mean']:.3f}+/-{r['cv_f1_std']:.3f} "
              f"(epochs={r['final_epochs']}, ckpt={r['checkpoint']})")

    # Confusion matrix for the best config.
    best = rows[0]
    best_preds = all_val_preds_by_cfg[best["cfg_id"]]
    cm = confusion_matrix(y, best_preds, labels=range(num_classes))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Stage 1 best (cfg_{best['cfg_id']:03d}): "
              f"acc={best['cv_acc_mean']:.3f} f1={best['cv_f1_mean']:.3f}")
    plt.tight_layout()
    out_path = OUT_DIR / "stage1_best_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved best-config confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
