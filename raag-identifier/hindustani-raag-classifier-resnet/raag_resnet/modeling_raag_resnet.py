# RaagResNetForAudioClassification
#
# Backbone: ported from carnatic-raga-classifier-jeevster's ResNetRagaClassifier
# (Conv1d stem (in_ch -> n_channel, k=80, stride) -> n_blocks x ResidualBlock(n_channel,
# k=3) each followed by MaxPool1d(2) -> global avg-pool over time). Module names
# (`conv_first`, `res_blocks`, `conv_block1`/`conv_block2`) match jeevster's
# `models.py` exactly so `ckpts/best_ckpt.tar`'s state_dict loads directly via
# `load_backbone_weights` (the original `fc1` is dropped).
#
# Head: small MLP (Linear -> [BatchNorm1d] -> ReLU -> [Dropout] -> ... -> Linear),
# matching the structure of probe_common.MLPHead so that
# `outputs/sweep/checkpoints/cfg_*.pt` (Stage 1) load directly via
# `load_head_weights`. `feat_mean`/`feat_scale` buffers hold the StandardScaler
# fit during Stage 1, applied to the backbone's pooled features before the head.

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_raag_resnet import RaagResNetConfig


def _conv_block(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding="same" if stride == 1 else 0),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


class _ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel_size=3):
        super().__init__()
        self.conv_block1 = _conv_block(n_channels, n_channels, kernel_size=kernel_size, stride=1)
        self.conv_block2 = _conv_block(n_channels, n_channels, kernel_size=3, stride=1)

    def forward(self, x):
        identity = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x + identity


def _build_head(in_dim, num_classes, hidden_dims, batchnorm, dropout):
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


class RaagResNetForAudioClassification(PreTrainedModel):
    config_class = RaagResNetConfig
    base_model_prefix = "raag_resnet"
    main_input_name = "input_values"

    def __init__(self, config: RaagResNetConfig):
        super().__init__(config)

        self.conv_first = _conv_block(
            config.backbone_input_channels, config.backbone_n_channel,
            kernel_size=80, stride=config.backbone_stride,
        )
        self.res_blocks = nn.ModuleList(
            _ResidualBlock(config.backbone_n_channel, kernel_size=3)
            for _ in range(config.backbone_n_blocks)
        )
        self.max_pool_every = config.backbone_max_pool_every

        # StandardScaler stats (fit on backbone features during head training).
        self.register_buffer("feat_mean", torch.zeros(config.backbone_n_channel))
        self.register_buffer("feat_scale", torch.ones(config.backbone_n_channel))

        self.head = _build_head(
            config.backbone_n_channel, config.num_labels,
            config.head_hidden_dims, config.head_batchnorm, config.head_dropout,
        )

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def backbone_forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """input_values: (batch, in_channels, T) -> (batch, backbone_n_channel)"""
        x = self.conv_first(input_values)
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i % self.max_pool_every == 0:
                x = F.max_pool1d(x, 2)
        x = F.avg_pool1d(x, x.shape[-1])
        return x.squeeze(-1)

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor = None) -> SequenceClassifierOutput:
        grad_ctx = torch.no_grad() if self.config.freeze_backbone else nullcontext()
        with grad_ctx:
            features = self.backbone_forward(input_values)

        features = (features - self.feat_mean) / self.feat_scale
        logits = self.head(features)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.conv_first.eval()
            self.res_blocks.eval()
        return self

    def load_backbone_weights(self, ckpt_path):
        """Load conv_first/res_blocks weights from jeevster's best_ckpt.tar (fc1 dropped)."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state"]
        backbone_sd = {
            k: v for k, v in state_dict.items()
            if k.startswith("conv_first.") or k.startswith("res_blocks.")
        }
        return self.load_state_dict(backbone_sd, strict=False)

    def load_head_weights(self, head_ckpt_path):
        """Load head weights + feature scaler from a Stage 1 outputs/sweep/checkpoints/cfg_*.pt."""
        ckpt = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
        head_sd = {k.removeprefix("net."): v for k, v in ckpt["state_dict"].items()}
        self.head.load_state_dict(head_sd)
        self.feat_mean.copy_(torch.as_tensor(ckpt["scaler_mean"], dtype=torch.float32))
        self.feat_scale.copy_(torch.as_tensor(ckpt["scaler_scale"], dtype=torch.float32))
        return ckpt["config"]
