"""The parts every architecture shares: tonic conditioning, the classifier head, and the
auxiliary swar-occupancy head.

The three backbones in this folder are wildly different -- a speech transformer, a 1-D
waveform ResNet, a 2-D spectrogram CNN -- but they all end in the same place: one pooled
feature vector per clip. Everything after that point is shared, so a change to how the tonic
is injected or how the DB prior is applied lands on all three at once and the comparison
between them stays honest.

    backbone -> (B, D) --[FiLM by tonic]--> (B, D) --> head --> (B, 50) logits
                                                  \\-> occupancy head -> (B, 12)
"""

import torch
import torch.nn as nn

from common.tonic import CONDITIONING_DIM


class FiLM(nn.Module):
    """Feature-wise linear modulation by the tonic: `h * (1 + gamma(t)) + beta(t)`.

    The alternative -- concatenating the tonic vector onto the feature -- lets the head use
    the tonic only additively. FiLM lets it *gate*, which is what the task actually needs:
    "this much energy 700 cents above Sa" means something different depending on where Sa
    is, and a gate can express that while a concatenated input cannot.

    Initialised to the identity (final layer zeroed), so a freshly built conditioned model
    is numerically identical to the unconditioned one and any difference in the training
    curve is the conditioning doing something rather than a different starting point.
    """

    def __init__(self, feature_dim, cond_dim=CONDITIONING_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, tonic):
        gamma, beta = self.net(tonic).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


def mlp_head(in_dim, out_dim, hidden=(256,), dropout=0.3, batchnorm=True):
    layers, dims = [], [in_dim, *hidden]
    for i in range(len(hidden)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if batchnorm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class RaagClassifier(nn.Module):
    """backbone + optional tonic FiLM + classifier head + optional occupancy head.

    tonic_mode : "none"       ignore the tonic vector entirely (the audio may still have
                              been normalised upstream -- that is the dataset's business)
                 "condition"  FiLM the pooled feature by it

    `aux_occupancy` adds a 12-way head predicting the raag's DB swar occupancy. It costs
    12*D parameters and is only used when the objective asks for it.
    """

    def __init__(self, backbone, feature_dim, num_labels=50, tonic_mode="none",
                 aux_occupancy=False, head_hidden=(256,), dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.tonic_mode = tonic_mode
        self.film = FiLM(feature_dim) if tonic_mode == "condition" else None
        self.head = mlp_head(feature_dim, num_labels, head_hidden, dropout)
        self.occupancy = nn.Linear(feature_dim, 12) if aux_occupancy else None

    def features(self, input_values):
        return self.backbone(input_values)

    def forward(self, input_values, tonic=None):
        h = self.features(input_values)
        if self.film is not None:
            if tonic is None:
                raise ValueError("tonic_mode='condition' but no tonic vector was supplied")
            h = self.film(h, tonic)
        out = {"logits": self.head(h), "features": h}
        if self.occupancy is not None:
            out["occupancy"] = self.occupancy(h)
        return out
