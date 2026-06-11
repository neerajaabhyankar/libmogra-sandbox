# Push the Stage 5b model (outputs/full50_warm_lastlayer_finetuned/model/) to the
# Hugging Face Hub as neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small.
#
# - Registers the custom config/model/feature-extractor classes for AutoClass loading
#   (adds `auto_map` to config.json / preprocessor_config.json) and copies the
#   raag_resnet/*.py source files alongside the weights, so
#   `from_pretrained(..., trust_remote_code=True)` works without any dependency on
#   this repo.
# - Writes a model card (README.md).
# - Assembles everything under outputs/full50_warm_lastlayer_finetuned/hf_push/ (new
#   dir, doesn't touch outputs/full50_warm_lastlayer_finetuned/model/) and uploads
#   that folder to the Hub.

import shutil
from pathlib import Path

from huggingface_hub import HfApi

from raag_resnet import RaagResNetConfig, RaagResNetForAudioClassification, RaagResNetFeatureExtractor

SOURCE_DIR = Path(__file__).parent / "outputs" / "full50_warm_lastlayer_finetuned" / "model"
PUSH_DIR = Path(__file__).parent / "outputs" / "full50_warm_lastlayer_finetuned" / "hf_push"
PKG_DIR = Path(__file__).parent / "raag_resnet"
REPO_ID = "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small"

README = """---
tags:
- audio-classification
- music
- hindustani-classical-music
- raga
datasets:
- neerajaabhyankar/hindustani-raag-small
pipeline_tag: audio-classification
---

# resnet-finetuned-1-hindustani-raag-small

A 1D convolutional ResNet that classifies the *raag* (melodic framework) of a Hindustani
classical music audio clip, across 50 raags.

## Model description

The backbone is a 10-block 1D residual network operating on raw audio (resampled to 8kHz,
2-channel). Its architecture and pretrained weights originate from a ResNet trained by
**jeevster** for Carnatic raga classification; we ported the backbone into this
self-contained `transformers`-compatible model (no dependency on the original training code)
and added a small classification head
(`Linear(300→64) → BatchNorm1d → ReLU → Dropout(0.2) → Linear(64→50)`).

## Training

We first trained the new head from scratch on frozen-backbone embeddings (a small
hyperparameter sweep over depth/width/batchnorm/dropout). Starting from that trained head, we
then unfroze only the backbone's last residual block and fine-tuned it jointly with the head,
end-to-end on raw audio (5-second random crops, batch size 8, discriminative learning rates),
keeping the rest of the backbone frozen. Training used the 1161-clip train split of
[`neerajaabhyankar/hindustani-raag-small`](https://huggingface.co/datasets/neerajaabhyankar/hindustani-raag-small)
(50 raags), with an 85/15 held-out validation split for early stopping.

## Evaluation

On the dataset's 92-clip test split: **accuracy 0.163, macro-F1 0.132** (chance level for 50
classes is 0.02).

## Usage

```python
import datasets
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model = AutoModelForAudioClassification.from_pretrained(
    "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small", trust_remote_code=True
)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small", trust_remote_code=True
)

ds = datasets.load_dataset("neerajaabhyankar/hindustani-raag-small", split="test")
audio = ds[0]["audio"]
inputs = feature_extractor(audio["array"], audio["sampling_rate"])
logits = model(inputs["input_values"].unsqueeze(0)).logits
predicted_raag = model.config.id2label[logits.argmax(-1).item()]
```
"""


def main():
    config = RaagResNetConfig.from_pretrained(SOURCE_DIR)
    model = RaagResNetForAudioClassification.from_pretrained(SOURCE_DIR)
    feature_extractor = RaagResNetFeatureExtractor.from_pretrained(SOURCE_DIR)

    RaagResNetConfig.register_for_auto_class()
    RaagResNetForAudioClassification.register_for_auto_class("AutoModelForAudioClassification")
    RaagResNetFeatureExtractor.register_for_auto_class("AutoFeatureExtractor")

    PUSH_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(PUSH_DIR)
    feature_extractor.save_pretrained(PUSH_DIR)

    for fname in ("configuration_raag_resnet.py", "modeling_raag_resnet.py", "feature_extraction_raag_resnet.py"):
        shutil.copy(PKG_DIR / fname, PUSH_DIR / fname)

    shutil.copy(SOURCE_DIR.parent / "test_confusion_matrix.png", PUSH_DIR / "test_confusion_matrix.png")

    (PUSH_DIR / "README.md").write_text(README)
    print(f"Assembled push directory: {PUSH_DIR}")
    print(sorted(p.name for p in PUSH_DIR.iterdir()))

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=PUSH_DIR, repo_id=REPO_ID, repo_type="model")
    print(f"Pushed to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
