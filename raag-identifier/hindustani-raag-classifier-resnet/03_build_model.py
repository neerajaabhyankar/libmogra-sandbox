# Stage 2: assemble RaagResNetForAudioClassification (see plan.md).
#
# - Backbone: conv_first + res_blocks, weights loaded from
#   carnatic-raga-classifier-jeevster/ckpts/best_ckpt.tar (fc1 dropped), frozen.
# - Head + feature scaler: loaded from outputs/sweep/checkpoints/cfg_030.pt, the
#   Stage 1 winner (depth=1, width=64, batchnorm=True, dropout=0.2).
# - Parity check: end-to-end backbone_forward(raw audio) vs the precomputed
#   `clip_mean` in embeddings-exploration/outputs/2s/crc-jeevster/, to confirm the
#   ported backbone is numerically identical to the original embedder.
# - End-to-end eval (train + test) -> confusion matrices.
# - Saves the assembled model + feature extractor (HF format, local only --
#   nothing is pushed) to outputs/model/.

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from probe_common import LABEL_NAMES, EMB_DIR, DATASET_ID, DATASET_REVISION
from raag_resnet import RaagResNetConfig, RaagResNetForAudioClassification, RaagResNetFeatureExtractor

JEEVSTER_CKPT = Path(__file__).parent.parent / "carnatic-raga-classifier-jeevster" / "ckpts" / "best_ckpt.tar"
HEAD_CKPT = Path(__file__).parent / "outputs" / "sweep" / "checkpoints" / "cfg_030.pt"
MODEL_DIR = Path(__file__).parent / "outputs" / "model"
EVAL_DIR = Path(__file__).parent / "outputs" / "model_eval"


def load_audio_split(split):
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)[split]
    samples = []
    for idx, ex in enumerate(ds):
        if ex["label"] not in range(len(LABEL_NAMES)):
            continue
        samples.append((idx, ex["audio"]["array"], ex["audio"]["sampling_rate"], ex["label"]))
    return samples


def evaluate(model, feature_extractor, samples, split, eval_dir):
    y_true, y_pred = [], []
    for _idx, array, sr, label in samples:
        inputs = feature_extractor(array, sr)["input_values"].unsqueeze(0)
        with torch.no_grad():
            logits = model(inputs).logits
        y_true.append(label)
        y_pred.append(logits.argmax(dim=-1).item())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"{split}: end-to-end accuracy = {acc:.3f} ({(y_true == y_pred).sum()}/{len(y_true)})")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABEL_NAMES)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Stage 2 end-to-end ({split}): acc={acc:.3f}")
    plt.tight_layout()
    out_path = eval_dir / f"stage2_{split}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"  saved {out_path}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    config = RaagResNetConfig(
        backbone_input_channels=2, backbone_n_channel=300, backbone_stride=16,
        backbone_n_blocks=10, backbone_max_pool_every=1,
        head_hidden_dims=[64], head_batchnorm=True, head_dropout=0.2,
        sampling_rate=8000, min_input_samples=40000,
        freeze_backbone=True,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )
    model = RaagResNetForAudioClassification(config)

    result = model.load_backbone_weights(JEEVSTER_CKPT)
    print(f"Backbone weights loaded from {JEEVSTER_CKPT.name}")
    print(f"  missing keys (expected: head.* + feat_mean/feat_scale): {len(result.missing_keys)}")
    print(f"  unexpected keys (expected: 0): {len(result.unexpected_keys)}")
    assert not result.unexpected_keys, result.unexpected_keys
    assert all(k.startswith("head.") or k in ("feat_mean", "feat_scale") for k in result.missing_keys)

    head_cfg = model.load_head_weights(HEAD_CKPT)
    print(f"Head + feature scaler loaded from {HEAD_CKPT.name}: {head_cfg}")

    model.eval()
    feature_extractor = RaagResNetFeatureExtractor(
        sampling_rate=config.sampling_rate, min_input_samples=config.min_input_samples,
    )

    print("\n--- Parity check vs precomputed crc-jeevster embeddings (first 5 train clips) ---")
    train_samples = load_audio_split("train")
    max_diff = 0.0
    for idx, array, sr, _label in train_samples[:5]:
        inputs = feature_extractor(array, sr)["input_values"].unsqueeze(0)
        with torch.no_grad():
            features = model.backbone_forward(inputs).squeeze(0).numpy()
        precomputed = np.load(EMB_DIR / f"train_{idx}.npz")["clip_mean"]
        diff = np.abs(features - precomputed).max()
        max_diff = max(max_diff, float(diff))
        print(f"  train_{idx}: max abs diff = {diff:.2e}")
    print(f"Max abs diff across checked clips: {max_diff:.2e}")

    print("\n--- End-to-end eval ---")
    evaluate(model, feature_extractor, train_samples, "train", EVAL_DIR)
    evaluate(model, feature_extractor, load_audio_split("test"), "test", EVAL_DIR)

    model.save_pretrained(MODEL_DIR)
    feature_extractor.save_pretrained(MODEL_DIR)
    print(f"\nSaved model + feature extractor to {MODEL_DIR} (not pushed anywhere)")


if __name__ == "__main__":
    main()
