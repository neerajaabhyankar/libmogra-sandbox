# Stage 3 (50-class): train the head on ALL 50 raag classes.
#
# - Loads outputs/embeddings_all/{train,test}.npz (from 04_compute_embeddings_50class.py)
# - Splits the 1161 train clips into train_inner / val (stratified, 85/15)
# - Trains the Stage 1 head architecture (cfg_030: Linear->BatchNorm->ReLU->Dropout->Linear,
#   width=64, dropout=0.2, wd=0.0, lr=1e-3) scaled to 50 classes, full-batch, with early
#   stopping on val loss
# - Saves training/val loss + macro-F1 curves
# - Assembles the full RaagResNetForAudioClassification (backbone + this head, num_labels=50)
#   and saves it (HF format, local only)
# - Evaluates on the official 92-clip test set -> accuracy, macro-F1, 50x50 confusion matrix
#
# Does not touch outputs/probe, outputs/sweep, or outputs/model (Stage 0-2 artifacts).

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from probe_common import DATASET_ID, DATASET_REVISION
from raag_resnet import RaagResNetConfig, RaagResNetForAudioClassification, RaagResNetFeatureExtractor

JEEVSTER_CKPT = Path(__file__).parent.parent / "carnatic-raga-classifier-jeevster" / "ckpts" / "best_ckpt.tar"
EMB_DIR = Path(__file__).parent / "outputs" / "embeddings_all"
OUT_DIR = Path(__file__).parent / "outputs" / "full50"
MODEL_DIR = OUT_DIR / "model"

SEED = 0
VAL_FRACTION = 0.15
EPOCHS = 300
PATIENCE = 30

# Head architecture: Stage 1 winner (cfg_030), scaled to 50 classes.
HIDDEN_DIMS = (64,)
BATCHNORM = True
DROPOUT = 0.2
WEIGHT_DECAY = 0.0
LR = 1e-3


def get_label_names():
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    return ds["train"].features["label"].names


def build_head(in_dim, num_classes):
    layers = []
    dims = [in_dim, *HIDDEN_DIMS]
    for i in range(len(HIDDEN_DIMS)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if BATCHNORM:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU())
        if DROPOUT > 0:
            layers.append(nn.Dropout(DROPOUT))
    layers.append(nn.Linear(dims[-1], num_classes))
    return nn.Sequential(*layers)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_npz = np.load(EMB_DIR / "train.npz")
    test_npz = np.load(EMB_DIR / "test.npz")
    X_all, y_all = train_npz["X"], train_npz["y"]
    X_test, y_test = test_npz["X"], test_npz["y"]

    label_names = get_label_names()
    num_classes = len(label_names)
    print(f"Train pool: {X_all.shape}, {num_classes} classes; test: {X_test.shape}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    train_idx, val_idx = next(sss.split(X_all, y_all))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"train_inner: {X_train.shape}, val: {X_val.shape}")

    scaler = StandardScaler().fit(X_train)
    Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    torch.manual_seed(SEED)
    head = build_head(X_all.shape[1], num_classes)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [],
               "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(EPOCHS):
        head.train()
        opt.zero_grad()
        logits = head(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            train_logits = head(Xt)
            train_loss = loss_fn(train_logits, yt).item()
            train_pred = train_logits.argmax(dim=1).numpy()

            val_logits = head(Xv)
            val_loss = loss_fn(val_logits, yv).item()
            val_pred = val_logits.argmax(dim=1).numpy()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(accuracy_score(y_train, train_pred))
        history["val_acc"].append(accuracy_score(y_val, val_pred))
        history["train_f1"].append(f1_score(y_train, train_pred, average="macro"))
        history["val_f1"].append(f1_score(y_val, val_pred, average="macro"))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= PATIENCE:
                break

    n_epochs_run = len(history["train_loss"])
    print(f"Stopped after {n_epochs_run} epochs, best epoch = {best_epoch} "
          f"(val_loss={best_val_loss:.4f}, val_acc={history['val_acc'][best_epoch]:.3f}, "
          f"val_f1={history['val_f1'][best_epoch]:.3f})")

    head.load_state_dict(best_state)
    head.eval()

    # --- Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs_axis = np.arange(n_epochs_run)
    axes[0].plot(epochs_axis, history["train_loss"], label="train")
    axes[0].plot(epochs_axis, history["val_loss"], label="val")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", label="best epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs_axis, history["train_f1"], label="train macro-F1")
    axes[1].plot(epochs_axis, history["val_f1"], label="val macro-F1")
    axes[1].plot(epochs_axis, history["train_acc"], label="train acc", linestyle=":")
    axes[1].plot(epochs_axis, history["val_acc"], label="val acc", linestyle=":")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", label="best epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Accuracy / macro-F1")
    axes[1].legend()

    fig.suptitle(f"Stage 3 (50 classes): training curves (best epoch={best_epoch})")
    fig.tight_layout()
    curves_path = OUT_DIR / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")

    # --- Save head-only checkpoint (consistent with Stage 1's cfg_*.pt format) ---
    head_ckpt_path = OUT_DIR / "head_checkpoint.pt"
    torch.save({
        "config": dict(depth=len(HIDDEN_DIMS), hidden_dims=HIDDEN_DIMS, batchnorm=BATCHNORM,
                        dropout=DROPOUT, weight_decay=WEIGHT_DECAY, lr=LR),
        "state_dict": head.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": history["val_acc"][best_epoch],
        "val_f1": history["val_f1"][best_epoch],
    }, head_ckpt_path)
    print(f"Saved {head_ckpt_path}")

    # --- Assemble full model and save (HF format, local only) ---
    config = RaagResNetConfig(
        backbone_input_channels=2, backbone_n_channel=300, backbone_stride=16,
        backbone_n_blocks=10, backbone_max_pool_every=1,
        head_hidden_dims=list(HIDDEN_DIMS), head_batchnorm=BATCHNORM, head_dropout=DROPOUT,
        sampling_rate=8000, min_input_samples=40000,
        freeze_backbone=True,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )
    model = RaagResNetForAudioClassification(config)
    result = model.load_backbone_weights(JEEVSTER_CKPT)
    assert not result.unexpected_keys, result.unexpected_keys

    model.head.load_state_dict(head.state_dict())
    model.feat_mean.copy_(torch.as_tensor(scaler.mean_, dtype=torch.float32))
    model.feat_scale.copy_(torch.as_tensor(scaler.scale_, dtype=torch.float32))
    model.eval()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    feature_extractor = RaagResNetFeatureExtractor(
        sampling_rate=config.sampling_rate, min_input_samples=config.min_input_samples,
    )
    feature_extractor.save_pretrained(MODEL_DIR)
    print(f"Saved assembled model + feature extractor to {MODEL_DIR}")

    # --- Test eval ---
    Xte = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    with torch.no_grad():
        test_logits = head(Xte)
    test_pred = test_logits.argmax(dim=1).numpy()

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    print(f"\nTest: accuracy={test_acc:.3f} ({(y_test == test_pred).sum()}/{len(y_test)}), "
          f"macro-F1={test_f1:.3f}")

    cm = confusion_matrix(y_test, test_pred, labels=range(num_classes))
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=label_names, yticklabels=label_names, ax=ax,
                annot_kws={"size": 6})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Stage 3 (50 classes) test confusion matrix: acc={test_acc:.3f} macro-F1={test_f1:.3f}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    cm_path = OUT_DIR / "test_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")


if __name__ == "__main__":
    main()
