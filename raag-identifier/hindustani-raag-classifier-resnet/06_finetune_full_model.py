# Stage 4 (50-class): full fine-tuning -- unfreeze the entire ResNet backbone
# (not just the head) and train end-to-end on raw audio, starting from the
# same jeevster checkpoint as before.
#
# - Loads the train split (1161 clips, 50 classes) only -- the 92-clip test
#   set is held out for the final eval, never used for training/early stopping.
# - Resamples every train clip to 8kHz mono once and caches it (pickle) so
#   reruns skip the slow resample step.
# - Splits into train_inner / val with the SAME StratifiedShuffleSplit(seed=0,
#   test_size=0.15) on the same (ordered) labels as Stage 3, so the split is
#   identical to outputs/full50.
# - Backbone (conv_first + res_blocks) initialized from
#   carnatic-raga-classifier-jeevster/ckpts/best_ckpt.tar (same starting point
#   as Stage 2/3); head is freshly initialized (Linear->BatchNorm->ReLU->
#   Dropout->Linear, width=64, dropout=0.2 -- same architecture as Stage 3).
# - config.freeze_backbone=False: backbone is trained too, with a discriminative
#   LR (backbone 1e-5, head 1e-3) since the pretrained ResNet was trained on a
#   different (Carnatic) repertoire and we have ~1000 clips.
# - Training uses random 5s crops (40000 samples @ 8kHz, batch_size=8) so
#   BatchNorm sees batch_size>1; val uses center crops of the same length (so
#   the BatchNorm running stats learned on 5s crops are evaluated on inputs of
#   the same scale). Final test eval uses full-length clips via
#   RaagResNetFeatureExtractor (matches the deployed preprocessing / Stage 2-3
#   methodology).
# - Saves training/val loss + macro-F1/accuracy curves, the fine-tuned model
#   (HF format, local only) and a 50x50 test confusion matrix.
#
# Does not touch outputs/probe, outputs/sweep, outputs/model, outputs/embeddings_all,
# or outputs/full50 (Stages 0-3 artifacts). Everything new goes to outputs/full50_finetuned.

import pickle
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from probe_common import DATASET_ID, DATASET_REVISION
from raag_resnet import RaagResNetConfig, RaagResNetForAudioClassification, RaagResNetFeatureExtractor

JEEVSTER_CKPT = Path(__file__).parent.parent / "carnatic-raga-classifier-jeevster" / "ckpts" / "best_ckpt.tar"
OUT_DIR = Path(__file__).parent / "outputs" / "full50_finetuned"
MODEL_DIR = OUT_DIR / "model"
WAVEFORM_CACHE = OUT_DIR / "train_waveforms_8k.pkl"

SEED = 0
VAL_FRACTION = 0.15
TARGET_SR = 8000
FIXED_LENGTH = 40000  # 5s @ 8kHz, matches config.min_input_samples
BATCH_SIZE = 8
EPOCHS = 40
PATIENCE = 8
GRAD_CLIP = 5.0

# Head architecture: same as Stage 3 (cfg_030, scaled to 50 classes), but
# freshly initialized here (full fine-tune starts the head from scratch).
HIDDEN_DIMS = (64,)
BATCHNORM = True
DROPOUT = 0.2

BACKBONE_LR = 1e-5
BACKBONE_WD = 1e-4
HEAD_LR = 1e-3
HEAD_WD = 0.0


def get_label_names():
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    return ds["train"].features["label"].names


def build_or_load_waveform_cache(label_names):
    if WAVEFORM_CACHE.exists():
        with open(WAVEFORM_CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded waveform cache from {WAVEFORM_CACHE} "
              f"({len(cache['waveforms'])} clips)")
        return cache["waveforms"], cache["labels"]

    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)["train"]

    waveforms, labels = [], []
    t0 = time.time()
    for i, ex in enumerate(ds):
        array, sr = ex["audio"]["array"], ex["audio"]["sampling_rate"]
        if sr != TARGET_SR:
            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        waveforms.append(array.astype(np.float32))
        labels.append(ex["label"])
        if (i + 1) % 200 == 0 or i == len(ds) - 1:
            print(f"  resampled {i + 1}/{len(ds)} ({time.time() - t0:.0f}s)")

    labels = np.array(labels, dtype=np.int64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(WAVEFORM_CACHE, "wb") as f:
        pickle.dump({"waveforms": waveforms, "labels": labels}, f)
    print(f"Saved waveform cache to {WAVEFORM_CACHE}")
    return waveforms, labels


class CroppedAudioDataset(Dataset):
    """Mono->stereo, fixed-length crop (random for train, center for val), per-channel normalize."""

    def __init__(self, waveforms, labels, indices, fixed_length, train):
        self.waveforms = waveforms
        self.labels = labels
        self.indices = indices
        self.fixed_length = fixed_length
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        wav = self.waveforms[idx]
        L = self.fixed_length
        if len(wav) < L:
            wav = np.pad(wav, (0, L - len(wav)))
        T = len(wav)
        if self.train:
            start = np.random.randint(0, T - L + 1)
        else:
            start = (T - L) // 2
        crop = wav[start:start + L]
        x = torch.from_numpy(crop).float().unsqueeze(0).repeat(2, 1)  # (2, L)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)
        return x, int(self.labels[idx])


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


def run_epoch(model, loader, opt, loss_fn, train):
    model.train(train)
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for x, y in loader:
            if train:
                opt.zero_grad()
            out = model(x, labels=y)
            if train:
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            bs = x.shape[0]
            total_loss += out.loss.item() * bs
            n += bs
            all_true.append(y.numpy())
            all_pred.append(out.logits.argmax(dim=1).numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": total_loss / n,
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    label_names = get_label_names()
    num_classes = len(label_names)
    print(f"{num_classes} classes")

    waveforms, labels = build_or_load_waveform_cache(label_names)
    print(f"Train pool: {len(waveforms)} clips")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    print(f"train_inner: {len(train_idx)}, val: {len(val_idx)}")

    train_ds = CroppedAudioDataset(waveforms, labels, train_idx, FIXED_LENGTH, train=True)
    val_ds = CroppedAudioDataset(waveforms, labels, val_idx, FIXED_LENGTH, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    config = RaagResNetConfig(
        backbone_input_channels=2, backbone_n_channel=300, backbone_stride=16,
        backbone_n_blocks=10, backbone_max_pool_every=1,
        head_hidden_dims=list(HIDDEN_DIMS), head_batchnorm=BATCHNORM, head_dropout=DROPOUT,
        sampling_rate=TARGET_SR, min_input_samples=FIXED_LENGTH,
        freeze_backbone=False,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )
    model = RaagResNetForAudioClassification(config)
    model.head = build_head(config.backbone_n_channel, num_classes)  # same shapes, fresh init
    result = model.load_backbone_weights(JEEVSTER_CKPT)
    assert not result.unexpected_keys, result.unexpected_keys
    print(f"Backbone weights loaded from {JEEVSTER_CKPT.name} (head freshly initialized)")

    backbone_params = list(model.conv_first.parameters()) + [
        p for block in model.res_blocks for p in block.parameters()
    ]
    head_params = list(model.head.parameters())
    opt = torch.optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": BACKBONE_WD},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": HEAD_WD},
    ])
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [],
               "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, opt, loss_fn, train=True)
        val_metrics = run_epoch(model, val_loader, opt, loss_fn, train=False)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"epoch {epoch:3d} ({time.time() - t0:5.1f}s)  "
              f"train: loss={train_metrics['loss']:.3f} acc={train_metrics['acc']:.3f} f1={train_metrics['f1']:.3f}  "
              f"val: loss={val_metrics['loss']:.3f} acc={val_metrics['acc']:.3f} f1={val_metrics['f1']:.3f}")

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for {PATIENCE} epochs)")
                break

    n_epochs_run = len(history["train_loss"])
    print(f"\nStopped after {n_epochs_run} epochs, best epoch = {best_epoch} "
          f"(val_loss={best_val_loss:.4f}, val_acc={history['val_acc'][best_epoch]:.3f}, "
          f"val_f1={history['val_f1'][best_epoch]:.3f})")

    model.load_state_dict(best_state)
    model.eval()

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

    fig.suptitle(f"Stage 4 (50 classes, full fine-tune): training curves (best epoch={best_epoch})")
    fig.tight_layout()
    curves_path = OUT_DIR / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    print(f"Saved {curves_path}")

    # --- Save full fine-tuned checkpoint ---
    ckpt_path = OUT_DIR / "finetuned_checkpoint.pt"
    torch.save({
        "config": dict(hidden_dims=HIDDEN_DIMS, batchnorm=BATCHNORM, dropout=DROPOUT,
                        backbone_lr=BACKBONE_LR, backbone_wd=BACKBONE_WD,
                        head_lr=HEAD_LR, head_wd=HEAD_WD, fixed_length=FIXED_LENGTH,
                        batch_size=BATCH_SIZE),
        "state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": history["val_acc"][best_epoch],
        "val_f1": history["val_f1"][best_epoch],
    }, ckpt_path)
    print(f"Saved {ckpt_path}")

    # --- Save assembled model (HF format, local only) ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    feature_extractor = RaagResNetFeatureExtractor(
        sampling_rate=config.sampling_rate, min_input_samples=config.min_input_samples,
    )
    feature_extractor.save_pretrained(MODEL_DIR)
    print(f"Saved fine-tuned model + feature extractor to {MODEL_DIR}")

    # --- Test eval (full-length clips, matches deployed feature extractor) ---
    from datasets import load_dataset
    test_ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)["test"]

    y_true, y_pred = [], []
    with torch.no_grad():
        for ex in test_ds:
            array, sr = ex["audio"]["array"], ex["audio"]["sampling_rate"]
            inputs = feature_extractor(array, sr)["input_values"].unsqueeze(0)
            logits = model(inputs).logits
            y_true.append(ex["label"])
            y_pred.append(logits.argmax(dim=-1).item())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nTest: accuracy={test_acc:.3f} ({(y_true == y_pred).sum()}/{len(y_true)}), "
          f"macro-F1={test_f1:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=label_names, yticklabels=label_names, ax=ax,
                annot_kws={"size": 6})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Stage 4 (50 classes, full fine-tune) test confusion matrix: "
                  f"acc={test_acc:.3f} macro-F1={test_f1:.3f}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    cm_path = OUT_DIR / "test_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"Saved {cm_path}")


if __name__ == "__main__":
    main()
