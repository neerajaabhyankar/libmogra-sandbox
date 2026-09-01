"""Architecture C -- a small 2-D ResNet over a Sa-anchored CQT.

This is the addition to the original brief, and the argument for it is structural.

For a waveform model the tonic is a nuisance parameter: the network has to learn, from 1810
clips, that the same raag sung with Sa at 101 Hz and at 289 Hz is the same thing. A CQT whose
`fmin` is the clip's own Sa has that invariance **built in** -- bin 0 is Sa, bin 3 is one
semitone above Sa, and the mapping from bin to swar is identical for every clip in the
corpus. Nothing is learned, nothing is resampled, nothing is interpolated.

Two design choices follow from what the representation is:

**Frequency resolution is preserved; time is pooled away.** Ordinary image CNNs pool both
axes, but here the frequency axis *is* the label -- which swars, and where the energy sits
between them. So the trunk pools time aggressively and frequency gently, and the final
pooling is over time only. What comes out is a (channels x frequency) profile: a learned,
multi-channel generalisation of exactly the pitch histogram that scores 0.40 in the sibling
project.

**Octave folding is available but off by default.** A raag is octave-invariant, so folding
the 4 octaves onto one is a correct prior -- but register is not *entirely* meaningless
(mandra-heavy alap versus taar-heavy taan), and the sibling project's M10 failed by
over-committing to hand-specified structure. Let the trunk see octaves; fold at the end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import RaagClassifier


class _Block(nn.Module):
    """Pre-activation residual block with a configurable (freq, time) pool."""

    def __init__(self, cin, cout, pool=(2, 2), dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.pool = pool
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = F.relu(h + self.skip(x))
        h = self.drop(h)
        return F.max_pool2d(h, self.pool) if self.pool != (1, 1) else h


class CQTBackbone(nn.Module):
    """(B, 1, n_bins, n_frames) log-CQT -> (B, proj_channels * n_bins_out) profile features.

    The output is deliberately *not* a global average: it keeps one value per (channel,
    frequency) cell, so the head can read "how much of channel k sits 700 cents above Sa".
    A global average pool over frequency would throw away the only axis that names a swar.
    """

    def __init__(self, n_bins=144, bins_per_octave=36, channels=(32, 64, 96, 128),
                 proj_channels=24, dropout=0.1, fold_octaves=False):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        self.fold_octaves = fold_octaves

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 5, padding=2, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(),
        )
        # pool time hard, frequency gently: (2,2) x3 then time-only, so 144 bins -> 18
        pools = [(2, 2), (2, 2), (2, 2), (1, 2)]
        blocks, cin = [], channels[0]
        for cout, pool in zip(channels, pools):
            blocks.append(_Block(cin, cout, pool=pool, dropout=dropout))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(cin, proj_channels, 1, bias=False)

        n_out = n_bins
        for pf, _pt in pools:
            n_out //= pf
        if fold_octaves:
            octaves = n_bins // bins_per_octave
            if n_out % octaves:
                raise ValueError(f"cannot fold {n_out} bins into {octaves} octaves")
            n_out //= octaves
        self.n_bins_out = n_out
        self.out_dim = proj_channels * n_out

    def forward(self, x):
        h = self.proj(self.blocks(self.stem(x)))     # (B, C, F', T')
        h = h.mean(dim=-1)                            # pool time only -> (B, C, F')
        if self.fold_octaves:
            b, c, f = h.shape
            octaves = f // self.n_bins_out
            h = h.reshape(b, c, octaves, self.n_bins_out).mean(dim=2)
        return h.flatten(1)


def backbone(n_bins=144, bins_per_octave=36, channels=(32, 64, 96, 128),
             proj_channels=24, dropout=0.1, fold_octaves=False, **_ignored):
    return CQTBackbone(n_bins=n_bins, bins_per_octave=bins_per_octave, channels=channels,
                       proj_channels=proj_channels, dropout=dropout,
                       fold_octaves=fold_octaves)


def build(num_labels=50, tonic_mode="none", aux_occupancy=False, n_bins=144,
          bins_per_octave=36, channels=(32, 64, 96, 128), proj_channels=24,
          dropout=0.1, fold_octaves=False, head_hidden=(256,), head_dropout=0.3,
          side_dim=0, side_out=64):
    backbone = CQTBackbone(n_bins=n_bins, bins_per_octave=bins_per_octave,
                           channels=channels, proj_channels=proj_channels,
                           dropout=dropout, fold_octaves=fold_octaves)
    return RaagClassifier(backbone, backbone.out_dim, num_labels=num_labels,
                          tonic_mode=tonic_mode, aux_occupancy=aux_occupancy,
                          head_hidden=head_hidden, dropout=head_dropout,
                          side_dim=side_dim, side_out=side_out)


def param_groups(model, lr=1e-3, head_lr=None, weight_decay=1e-4):
    """One group: this backbone is trained from scratch, so there is nothing to protect."""
    return [{"params": [p for p in model.parameters() if p.requires_grad],
             "lr": lr, "weight_decay": weight_decay}]
