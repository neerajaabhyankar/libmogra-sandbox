# Config for RaagResNetForAudioClassification: jeevster's 1D-ResNet backbone
# (carnatic-raga-classifier-jeevster, ckpts/best_ckpt.tar / config0.yaml) + a small
# MLP head trained on Hindustani raag labels (see ../plan.md, Stage 1 sweep).

from transformers import PretrainedConfig


class RaagResNetConfig(PretrainedConfig):
    model_type = "raag_resnet"

    def __init__(
        self,
        # Backbone hyperparams -- must match carnatic-raga-classifier-jeevster/config0.yaml
        # for ckpts/best_ckpt.tar to load.
        backbone_input_channels=2,
        backbone_n_channel=300,
        backbone_stride=16,
        backbone_n_blocks=10,
        backbone_max_pool_every=1,
        # Head hyperparams (Stage 1 best config, cfg_030: depth=1, width=64,
        # batchnorm=True, dropout=0.2).
        head_hidden_dims=(64,),
        head_batchnorm=True,
        head_dropout=0.2,
        # Audio preprocessing (matches embeddings-exploration/models/crc_jeevster.py).
        sampling_rate=8000,
        min_input_samples=40000,
        # If True, the backbone is frozen (no grad, kept in eval() mode) and only
        # the head is trainable.
        freeze_backbone=True,
        **kwargs,
    ):
        self.backbone_input_channels = backbone_input_channels
        self.backbone_n_channel = backbone_n_channel
        self.backbone_stride = backbone_stride
        self.backbone_n_blocks = backbone_n_blocks
        self.backbone_max_pool_every = backbone_max_pool_every
        self.head_hidden_dims = list(head_hidden_dims)
        self.head_batchnorm = head_batchnorm
        self.head_dropout = head_dropout
        self.sampling_rate = sampling_rate
        self.min_input_samples = min_input_samples
        self.freeze_backbone = freeze_backbone
        super().__init__(**kwargs)
