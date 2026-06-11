# Stage 3 (50-class): compute crc-jeevster backbone embeddings for the FULL dataset
# (all 50 raag classes), via the Stage 2 RaagResNetForAudioClassification backbone --
# verified numerically identical to embeddings-exploration's crc-jeevster embedder
# (see plan.md, Stage 2 parity check).
#
# Does NOT touch embeddings-exploration/outputs (which only covers labels 0-4).
# Saves one .npz per split with all embeddings + labels:
#   outputs/embeddings_all/{train,test}.npz  -- keys: X (N,300), y (N,), idx (N,)

from pathlib import Path

import numpy as np
import torch

from probe_common import DATASET_ID, DATASET_REVISION
from raag_resnet import RaagResNetConfig, RaagResNetForAudioClassification, RaagResNetFeatureExtractor

JEEVSTER_CKPT = Path(__file__).parent.parent / "carnatic-raga-classifier-jeevster" / "ckpts" / "best_ckpt.tar"
OUT_DIR = Path(__file__).parent / "outputs" / "embeddings_all"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = RaagResNetConfig(freeze_backbone=True)  # head/num_labels irrelevant here
    model = RaagResNetForAudioClassification(config)
    result = model.load_backbone_weights(JEEVSTER_CKPT)
    assert not result.unexpected_keys, result.unexpected_keys
    model.eval()

    feature_extractor = RaagResNetFeatureExtractor(
        sampling_rate=config.sampling_rate, min_input_samples=config.min_input_samples,
    )

    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)

    for split in ("train", "test"):
        n = len(ds[split])
        X = np.zeros((n, config.backbone_n_channel), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for idx, ex in enumerate(ds[split]):
            array, sr = ex["audio"]["array"], ex["audio"]["sampling_rate"]
            inputs = feature_extractor(array, sr)["input_values"].unsqueeze(0)
            with torch.no_grad():
                X[idx] = model.backbone_forward(inputs).squeeze(0).numpy()
            y[idx] = ex["label"]
            if (idx + 1) % 100 == 0 or idx == n - 1:
                print(f"  [{split}] {idx + 1}/{n}")

        out_path = OUT_DIR / f"{split}.npz"
        np.savez(out_path, X=X, y=y, idx=np.arange(n))
        print(f"Saved {out_path} -- X{X.shape}, {len(np.unique(y))} classes")


if __name__ == "__main__":
    main()
