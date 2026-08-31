# Stage 5b (50-class): warm-started fine-tune of only the LAST backbone layer.
#
# Stage 3 froze the *entire* backbone (conv_first + all 10 res_blocks) and only
# trained a brand-new head on top of fixed embeddings. In jeevster's original
# architecture the backbone's last residual block feeds straight into
# `fc1: Linear(300->150)` (which Stage 2/3 dropped and replaced with our own
# head) -- so "the last layer" here means res_blocks[9] (the 10th/last residual
# block), the block immediately before the global-avg-pool that produces the
# 300-dim embedding our head consumes.
#
# This stage: warm-start the head + feature scaler from Stage 3
# (outputs/full50/head_checkpoint.pt, same as 07), keep conv_first and
# res_blocks[0..8] frozen (eval mode, requires_grad=False, BN running stats
# untouched), and unfreeze only res_blocks[9] + the head. This is a much
# smaller trainable surface than Stage 4/5a (~0.6M vs ~7M params) so it should
# be both faster per epoch and more stable.
#
# Train/val split, crop strategy, and eval methodology match Stage 4/5a.
#
# Does not touch outputs/probe, outputs/sweep, outputs/model, outputs/embeddings_all,
# outputs/full50, outputs/full50_finetuned, or outputs/full50_warm_finetuned
# (Stages 0-5a artifacts). Everything new goes to outputs/full50_warm_lastlayer_finetuned.

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from finetune_common import (
    FIXED_LENGTH, get_label_names, build_or_load_waveform_cache,
    CroppedAudioDataset, run_epoch, plot_curves, eval_test_confusion_matrix,
)
from raag_resnet import RaagResNetConfig, RaagResNetForAudioClassification, RaagResNetFeatureExtractor

JEEVSTER_CKPT = Path(__file__).parent.parent / "carnatic-raga-classifier-jeevster" / "ckpts" / "best_ckpt.tar"
STAGE3_HEAD_CKPT = Path(__file__).parent / "outputs" / "full50" / "head_checkpoint.pt"
CACHE_DIR = Path(__file__).parent / "outputs" / "_cache"
OUT_DIR = Path(__file__).parent / "outputs" / "full50_warm_lastlayer_finetuned"
MODEL_DIR = OUT_DIR / "model"

SEED = 0
VAL_FRACTION = 0.15
BATCH_SIZE = 8
EPOCHS = 40
PATIENCE = 10
GRAD_CLIP = 5.0

# Head architecture must match Stage 3's (so load_head_weights' shapes line up).
HIDDEN_DIMS = (64,)
BATCHNORM = True
DROPOUT = 0.2

LAST_BLOCK_LR = 1e-4
LAST_BLOCK_WD = 1e-4
HEAD_LR = 1e-4
HEAD_WD = 0.0


def set_mode(model, train):
    model.train(train)
    if train:
        # keep the frozen part of the backbone in eval mode (BN running stats untouched)
        model.conv_first.eval()
        for i in range(len(model.res_blocks) - 1):
            model.res_blocks[i].eval()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    label_names = get_label_names()
    num_classes = len(label_names)
    print(f"{num_classes} classes")

    waveforms, labels = build_or_load_waveform_cache(CACHE_DIR / "train_waveforms_8k.pkl")
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
        sampling_rate=8000, min_input_samples=FIXED_LENGTH,
        freeze_backbone=False,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )
    model = RaagResNetForAudioClassification(config)
    result = model.load_backbone_weights(JEEVSTER_CKPT)
    assert not result.unexpected_keys, result.unexpected_keys
    head_cfg = model.load_head_weights(STAGE3_HEAD_CKPT)
    print(f"Backbone weights from {JEEVSTER_CKPT.name}; head + feature scaler "
          f"warm-started from {STAGE3_HEAD_CKPT} ({head_cfg})")

    # Freeze everything except the last residual block + head.
    for p in model.conv_first.parameters():
        p.requires_grad = False
    for block in model.res_blocks[:-1]:
        for p in block.parameters():
            p.requires_grad = False

    last_block_params = list(model.res_blocks[-1].parameters())
    head_params = list(model.head.parameters())
    n_trainable = sum(p.numel() for p in last_block_params + head_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} (last res_block + head)")

    opt = torch.optim.Adam([
        {"params": last_block_params, "lr": LAST_BLOCK_LR, "weight_decay": LAST_BLOCK_WD},
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
        train_metrics = run_epoch(model, train_loader, opt, loss_fn, train=True,
                                   set_mode=set_mode, grad_clip=GRAD_CLIP)
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

    plot_curves(history, best_epoch,
                "Stage 5b (50 classes, warm-started last-layer fine-tune): training curves",
                OUT_DIR / "training_curves.png")

    ckpt_path = OUT_DIR / "finetuned_checkpoint.pt"
    torch.save({
        "config": dict(hidden_dims=HIDDEN_DIMS, batchnorm=BATCHNORM, dropout=DROPOUT,
                        last_block_lr=LAST_BLOCK_LR, last_block_wd=LAST_BLOCK_WD,
                        head_lr=HEAD_LR, head_wd=HEAD_WD, fixed_length=FIXED_LENGTH,
                        batch_size=BATCH_SIZE, warm_start_head=str(STAGE3_HEAD_CKPT)),
        "state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": history["val_acc"][best_epoch],
        "val_f1": history["val_f1"][best_epoch],
    }, ckpt_path)
    print(f"Saved {ckpt_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    feature_extractor = RaagResNetFeatureExtractor(
        sampling_rate=config.sampling_rate, min_input_samples=config.min_input_samples,
    )
    feature_extractor.save_pretrained(MODEL_DIR)
    print(f"Saved fine-tuned model + feature extractor to {MODEL_DIR}")

    eval_test_confusion_matrix(
        model, feature_extractor, label_names,
        "Stage 5b (50 classes, warm-started last-layer fine-tune) test confusion matrix",
        OUT_DIR / "test_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
