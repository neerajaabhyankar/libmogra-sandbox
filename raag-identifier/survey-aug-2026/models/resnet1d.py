"""Architecture R -- the jeevster 1-D waveform ResNet, warm-started from the Carnatic
raga classifier.

The backbone is `carnatic-raga-classifier-jeevster`: a Conv1d stem (2 -> 300 channels,
kernel 80, stride 16) followed by 10 residual blocks with max-pooling, global-average-pooled
to 300 dims. Pretrained on 150 Carnatic ragas at 8 kHz stereo. We import the module from
`../hindustani-raag-classifier-resnet/raag_resnet/` rather than re-porting it, so the two
projects cannot drift on what the architecture is.

What the sibling project learned about fine-tuning it, which sets the default here:

    full unfreeze, random head        0.054   the head destroys the backbone
    full unfreeze, warm-started head  0.098
    frozen backbone, new head         0.109
    last block + head                 0.163
    **last two blocks + head**        0.174   best

So `unfreeze_blocks=2` is the default. Those numbers are v0 test accuracy and are not
directly comparable to anything in this folder -- the ranking is what carries over.
"""

import torch
import torch.nn as nn

from common.paths import JEEVSTER_DIR, add_sibling_paths

from .heads import RaagClassifier


class JeevsterBackbone(nn.Module):
    """(B, 2, T) waveform at 8 kHz -> (B, 300) globally average-pooled features."""

    def __init__(self, checkpoint=None, unfreeze_blocks=2):
        super().__init__()
        add_sibling_paths()
        from raag_resnet.configuration_raag_resnet import RaagResNetConfig
        from raag_resnet.modeling_raag_resnet import RaagResNetForAudioClassification

        cfg = RaagResNetConfig(num_labels=50, freeze_backbone=False)
        host = RaagResNetForAudioClassification(cfg)
        ckpt = checkpoint or (JEEVSTER_DIR / "ckpts" / "best_ckpt.tar")
        missing = host.load_backbone_weights(ckpt)
        if getattr(missing, "unexpected_keys", None):
            raise RuntimeError(f"unexpected keys loading {ckpt}: {missing.unexpected_keys}")

        self.conv_first = host.conv_first
        self.res_blocks = host.res_blocks
        self.max_pool_every = host.max_pool_every
        self.out_dim = cfg.backbone_n_channel
        self.set_trainable(unfreeze_blocks)

    def set_trainable(self, unfreeze_blocks):
        """Freeze everything, then thaw the last `unfreeze_blocks` residual blocks.
        `unfreeze_blocks=0` is a frozen feature extractor; `-1` unfreezes everything."""
        for p in self.parameters():
            p.requires_grad = False
        if unfreeze_blocks == -1:
            for p in self.parameters():
                p.requires_grad = True
        elif unfreeze_blocks > 0:
            for blk in self.res_blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        self.unfrozen = unfreeze_blocks

    def train(self, mode=True):
        """Keep frozen BatchNorms in eval mode -- otherwise their running statistics drift
        on this corpus even though their weights cannot, which quietly changes a "frozen"
        backbone's output between epochs."""
        super().train(mode)
        if self.unfrozen == 0:
            self.conv_first.eval()
            self.res_blocks.eval()
        elif self.unfrozen > 0:
            self.conv_first.eval()
            for blk in self.res_blocks[:len(self.res_blocks) - self.unfrozen]:
                blk.eval()
        return self

    def forward(self, x):
        import torch.nn.functional as F

        h = self.conv_first(x)
        for i, block in enumerate(self.res_blocks):
            h = block(h)
            if i % self.max_pool_every == 0:
                h = F.max_pool1d(h, 2)
        return F.avg_pool1d(h, h.shape[-1]).squeeze(-1)


def backbone(unfreeze_blocks=2, checkpoint=None, **_ignored):
    return JeevsterBackbone(checkpoint, unfreeze_blocks=unfreeze_blocks)


def build(num_labels=50, tonic_mode="none", aux_occupancy=False, unfreeze_blocks=2,
          head_hidden=(64,), dropout=0.2, checkpoint=None, side_dim=0, side_out=64):
    backbone = JeevsterBackbone(checkpoint, unfreeze_blocks=unfreeze_blocks)
    return RaagClassifier(backbone, backbone.out_dim, num_labels=num_labels,
                          tonic_mode=tonic_mode, aux_occupancy=aux_occupancy,
                          head_hidden=head_hidden, dropout=dropout,
                          side_dim=side_dim, side_out=side_out)


def param_groups(model, lr=1e-4, head_lr=None, weight_decay=1e-4):
    head_lr = head_lr if head_lr is not None else lr
    head_params, back_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith(("head.", "film.", "side.", "occupancy.")) else back_params).append(p)
    groups = [{"params": head_params, "lr": head_lr, "weight_decay": 0.0}]
    if back_params:
        groups.append({"params": back_params, "lr": lr, "weight_decay": weight_decay})
    return groups
