# Shared helpers for the Stage 4+ fine-tuning scripts (06/07/08). Does not modify
# raag_resnet/ or probe_common.py -- new shared code only.

import pickle
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from probe_common import DATASET_ID, DATASET_REVISION

TARGET_SR = 8000
FIXED_LENGTH = 40000  # 5s @ 8kHz, matches config.min_input_samples


def get_label_names():
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    return ds["train"].features["label"].names


def build_or_load_waveform_cache(cache_path: Path):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded waveform cache from {cache_path} ({len(cache['waveforms'])} clips)")
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"waveforms": waveforms, "labels": labels}, f)
    print(f"Saved waveform cache to {cache_path}")
    return waveforms, labels


class CroppedAudioDataset(Dataset):
    """Mono->stereo, fixed-length crop (random for train, center for val), per-channel normalize."""

    def __init__(self, waveforms, labels, indices, fixed_length=FIXED_LENGTH, train=True):
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


def build_head(in_dim, num_classes, hidden_dims=(64,), batchnorm=True, dropout=0.2):
    layers = []
    dims = [in_dim, *hidden_dims]
    for i in range(len(hidden_dims)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if batchnorm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], num_classes))
    return nn.Sequential(*layers)


def run_epoch(model, loader, opt, loss_fn, train, set_mode=None, grad_clip=5.0):
    if set_mode is None:
        model.train(train)
    else:
        set_mode(model, train)

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
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], grad_clip,
                )
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


def plot_curves(history, best_epoch, title, out_path):
    n_epochs_run = len(history["train_loss"])
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

    fig.suptitle(f"{title} (best epoch={best_epoch})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def eval_test_confusion_matrix(model, feature_extractor, label_names, title, out_path):
    from datasets import load_dataset
    test_ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)["test"]
    num_classes = len(label_names)

    model.eval()
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
    ax.set_title(f"{title}: acc={test_acc:.3f} macro-F1={test_f1:.3f}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    return test_acc, test_f1
