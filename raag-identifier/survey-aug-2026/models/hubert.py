"""Architecture D -- distilHuBERT (`ntu-spml/distilhubert`), the model the original
notebook fine-tuned.

Two things differ from `../hindustani-raag-identifier.ipynb`, both forced by measurement
rather than taste:

**The convolutional feature encoder is frozen.** Benchmarked on the M1: unfrozen, a 20 s
batch of 8 takes 87 s/step and thrashes memory; frozen, 3.1 s/step. That is 5.5 hours per
epoch against 12 minutes. Freezing the CNN front end is also the standard recipe for HuBERT
fine-tuning, so nothing is given up. `--unfreeze-encoder` exists for a GPU.

**Mean pooling over time, not the library's default projector.** `HubertForSequenceClassification`
runs a projector then mean-pools, which is fine, but this project needs the pooled feature
itself as the seam where FiLM conditioning and the auxiliary head attach. Same computation,
one hook.
"""

import torch
import torch.nn as nn

from .heads import RaagClassifier

MODEL_ID = "ntu-spml/distilhubert"


class HubertBackbone(nn.Module):
    """(B, T) waveform at 16 kHz -> (B, 768) mean-pooled last hidden state."""

    def __init__(self, model_id=MODEL_ID, freeze_encoder=True, freeze_all=False):
        super().__init__()
        from transformers import AutoModel

        self.hubert = AutoModel.from_pretrained(model_id)
        self.out_dim = self.hubert.config.hidden_size
        if freeze_encoder:
            self.hubert.feature_extractor._freeze_parameters()
        if freeze_all:
            for p in self.hubert.parameters():
                p.requires_grad = False

    def forward(self, x):
        h = self.hubert(x).last_hidden_state          # (B, frames, 768)
        return h.mean(dim=1)


def backbone(freeze_encoder=True, freeze_all=False, model_id=MODEL_ID, **_ignored):
    return HubertBackbone(model_id, freeze_encoder=freeze_encoder, freeze_all=freeze_all)


def build(num_labels=50, tonic_mode="none", aux_occupancy=False, freeze_encoder=True,
          freeze_all=False, head_hidden=(256,), dropout=0.3, model_id=MODEL_ID):
    backbone = HubertBackbone(model_id, freeze_encoder=freeze_encoder, freeze_all=freeze_all)
    return RaagClassifier(backbone, backbone.out_dim, num_labels=num_labels,
                          tonic_mode=tonic_mode, aux_occupancy=aux_occupancy,
                          head_hidden=head_hidden, dropout=dropout)


def param_groups(model, lr=1e-4, head_lr=None, weight_decay=1e-4):
    """Backbone and head on separate learning rates.

    A randomly initialised head backpropagating into pretrained weights at the same LR
    wrecks them in the first few hundred steps -- that is visible in the sibling ResNet
    project, where a full unfreeze with a random head scored 0.054 against 0.174 for the
    warm-started, partially-frozen version. Head at 10x by default.
    """
    head_lr = head_lr if head_lr is not None else lr * 10
    head_params, back_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith(("head.", "film.", "occupancy.")) else back_params).append(p)
    groups = [{"params": head_params, "lr": head_lr, "weight_decay": 0.0}]
    if back_params:
        groups.append({"params": back_params, "lr": lr, "weight_decay": weight_decay})
    return groups
