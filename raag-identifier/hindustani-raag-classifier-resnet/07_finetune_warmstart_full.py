# Stage 5a (50-class): full fine-tune, but WARM-STARTED.
#
# Same as Stage 4 (06_finetune_full_model.py) -- unfreeze the entire backbone
# (conv_first + all 10 res_blocks) and the head, train end-to-end on raw audio,
# train split only -- EXCEPT the head + feature scaler are initialized from
# Stage 3's trained head (outputs/full50/head_checkpoint.pt) instead of being
# randomly initialized. Stage 4 found that a random head produced huge/unstable
# early gradients and never recovered within a tractable epoch budget; warm-
# starting from an already-trained head should make early training much more
# stable.
#
# Train/val split, crop strategy, optimizer structure (discriminative LRs),
# and eval methodology all match Stage 4. Only the initialization and a couple
# of LR/patience knobs differ (see constants below).
#
# Does not touch outputs/probe, outputs/sweep, outputs/model, outputs/embeddings_all,
# outputs/full50, or outputs/full50_finetuned (Stages 0-4 artifacts). Everything
# new goes to outputs/full50_warm_finetuned.

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from finetune_common import (
    FIXED_LENGTH, get_label_names, build_or_load_waveform_cache,
    CroppedAudioDataset, build_head, run_epoch, plot_curves, eval_test_confusion_matrix,
)
from raag_resnet import RaagResNetConfig, RaagResNetForAudioClassification, RaagResNetFeatureExtractor

JEEVSTER_CKPT = Path(__file__).parent.parent / "carnatic-raga-classifier-jeevster" / "ckpts" / "best_ckpt.tar"
STAGE3_HEAD_CKPT = Path(__file__).parent / "outputs" / "full50" / "head_checkpoint.pt"
CACHE_DIR = Path(__file__).parent / "outputs" / "_cache"
OUT_DIR = Path(__file__).parent / "outputs" / "full50_warm_finetuned"
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

# Lower LRs than Stage 4 -- the head is already trained, so we want gentle
# adaptation rather than re-randomizing it. Backbone LR raised slightly (3e-5
# vs Stage 4's 1e-5) since stable head gradients should make this safer.
BACKBONE_LR = 3e-5
BACKBONE_WD = 1e-4
HEAD_LR = 1e-4
HEAD_WD = 0.0


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

    import time
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, opt, loss_fn, train=True, grad_clip=GRAD_CLIP)
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

    plot_curves(history, best_epoch, "Stage 5a (50 classes, warm-started full fine-tune): training curves",
                OUT_DIR / "training_curves.png")

    ckpt_path = OUT_DIR / "finetuned_checkpoint.pt"
    torch.save({
        "config": dict(hidden_dims=HIDDEN_DIMS, batchnorm=BATCHNORM, dropout=DROPOUT,
                        backbone_lr=BACKBONE_LR, backbone_wd=BACKBONE_WD,
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
        "Stage 5a (50 classes, warm-started full fine-tune) test confusion matrix",
        OUT_DIR / "test_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
