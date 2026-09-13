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

**How time is summarised is a choice, `pool`.** The trunk's receptive field is 80 frames,
3.7 s, so each of the ~26 positions left at the end already encodes a few swars *in order*.
What the default mean then discards is the order *between* those 3.7 s stretches:

    mean    average over time. Order-free beyond 3.7 s. The original.
    stats   mean and standard deviation. Adds how much each feature varies; still order-free.
    attn    a learned weighting of positions. Picks *which* moments count; still order-free.
    tconv   three dilated temporal convolutions (1, 2, 4) over the position sequence, added
            residually, then the mean -- the receptive field grows to ~14 s of the 20 s
            window. Order-sensitive and shift-invariant. Starts as exactly `mean`.
    gru     a bidirectional GRU over the positions, its mean concatenated with `mean`'s.
            Order-sensitive over the whole window.

`stats` and `attn` are the controls: if `tconv` or `gru` win and they do not, order is what
helped; if all four win equally, it was only a richer summary.

A layer whose kernel spans the whole time axis is deliberately *not* on the list: it assigns
a separate weight to each absolute position in the window, and a 20 s crop starting at a
random point in a performance has no meaningful absolute positions -- a phrase at second 3
and the same phrase at second 13 would be different features.

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


POOLS = ("mean", "stats", "attn", "tconv", "gru")


class TimePool(nn.Module):
    """(B, C, F, T) -> (B, out_dim): how the time axis is summarised. See the module doc."""

    def __init__(self, kind, feature_dim, hidden=128):
        super().__init__()
        if kind not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {kind!r}")
        self.kind = kind
        d = feature_dim
        self.out_dim = {"stats": 2 * d, "gru": d + 2 * hidden}.get(kind, d)
        if kind == "attn":
            self.score = nn.Sequential(nn.Conv1d(d, hidden, 1), nn.Tanh(),
                                       nn.Conv1d(hidden, 1, 1))
        elif kind == "tconv":
            layers, cin = [], d
            for dilation in (1, 2, 4):
                layers += [nn.Conv1d(cin, hidden, 3, padding=dilation, dilation=dilation),
                           nn.BatchNorm1d(hidden), nn.ReLU()]
                cin = hidden
            layers.append(nn.Conv1d(hidden, d, 1))
            self.temporal = nn.Sequential(*layers)
            # zero-initialised residual branch: epoch 0 is exactly the mean-pooled model
            nn.init.zeros_(self.temporal[-1].weight)
            nn.init.zeros_(self.temporal[-1].bias)
        elif kind == "gru":
            self.gru = nn.GRU(d, hidden, batch_first=True, bidirectional=True)

    def forward(self, h):
        x = h.flatten(1, 2)                                  # (B, C*F, T)
        if self.kind == "mean":
            return x.mean(-1)
        if self.kind == "stats":
            return torch.cat([x.mean(-1), x.std(-1)], dim=1)
        if self.kind == "attn":
            return (x * torch.softmax(self.score(x), dim=-1)).sum(-1)
        if self.kind == "tconv":
            return (x + self.temporal(x)).mean(-1)
        seq, _ = self.gru(x.transpose(1, 2))
        return torch.cat([x.mean(-1), seq.mean(1)], dim=1)


class CQTBackbone(nn.Module):
    """(B, 1, n_bins, n_frames) log-CQT -> (B, proj_channels * n_bins_out) profile features.

    The output is deliberately *not* a global average: it keeps one value per (channel,
    frequency) cell, so the head can read "how much of channel k sits 700 cents above Sa".
    A global average pool over frequency would throw away the only axis that names a swar.
    """

    def __init__(self, n_bins=144, bins_per_octave=36, channels=(32, 64, 96, 128),
                 proj_channels=24, dropout=0.1, fold_octaves=False, pool="mean",
                 freq_pools=3):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        self.fold_octaves = fold_octaves
        if fold_octaves and pool != "mean":
            raise ValueError("fold_octaves is only defined for pool='mean'")

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 5, padding=2, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(),
        )
        # pool time hard, frequency gently: frequency halves in the first `freq_pools`
        # blocks only, time in every block. The default (3) leaves 144 bins -> 18 cells,
        # ~2.7 semitones each -- coarser than the swar grid, so sub-cell pitch has to live
        # in the channels. 2 -> 36 cells, 1 -> 72 (~67 cents). At depth 4 and 3 frequency
        # pools this is (2,2) x3 then (1,2), exactly the original.
        pools = [(2 if i < freq_pools else 1, 2) for i in range(len(channels))]
        blocks, cin = [], channels[0]
        for cout, block_pool in zip(channels, pools):
            blocks.append(_Block(cin, cout, pool=block_pool, dropout=dropout))
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
        self.proj_channels = proj_channels
        # the pooling sees the un-folded frequency axis; folding, when on, happens after
        f_pooled = n_out * (n_bins // bins_per_octave if fold_octaves else 1)
        self.time_pool = TimePool(pool, proj_channels * f_pooled)
        self.out_dim = proj_channels * n_out if fold_octaves else self.time_pool.out_dim

    def forward(self, x):
        h = self.proj(self.blocks(self.stem(x)))     # (B, C, F', T')
        z = self.time_pool(h)                         # (B, out_dim)
        if self.fold_octaves:
            c = self.proj_channels
            f = z.shape[1] // c
            z = z.reshape(-1, c, f // self.n_bins_out, self.n_bins_out).mean(dim=2).flatten(1)
        return z


#: The trunk's channel widths, block by block. `depth` takes a prefix, `width` scales it.
BASE_CHANNELS = (32, 64, 96, 128, 160, 192)
BASE_PROJ = 24


def shape(width=1.0, depth=4):
    """(channels, proj_channels) for a width multiplier and a block count.
    width=1, depth=4 is the original (32, 64, 96, 128) with a 24-channel projection."""
    if not 1 <= depth <= len(BASE_CHANNELS):
        raise ValueError(f"depth must be 1..{len(BASE_CHANNELS)}")
    return (tuple(max(4, int(round(c * width))) for c in BASE_CHANNELS[:depth]),
            max(4, int(round(BASE_PROJ * width))))


def backbone(n_bins=144, bins_per_octave=36, dropout=0.1, fold_octaves=False,
             pool="mean", width=1.0, depth=4, freq_pools=3, **_ignored):
    channels, proj = shape(width, depth)
    return CQTBackbone(n_bins=n_bins, bins_per_octave=bins_per_octave, channels=channels,
                       proj_channels=proj, dropout=dropout, fold_octaves=fold_octaves,
                       pool=pool, freq_pools=freq_pools)


def build(num_labels=50, tonic_mode="none", aux_occupancy=False, n_bins=144,
          bins_per_octave=36, dropout=0.1, fold_octaves=False, pool="mean", width=1.0,
          depth=4, freq_pools=3, head_hidden=(256,), head_dropout=0.3, side_dim=0,
          side_out=64):
    trunk = backbone(n_bins=n_bins, bins_per_octave=bins_per_octave, dropout=dropout,
                     fold_octaves=fold_octaves, pool=pool, width=width, depth=depth,
                     freq_pools=freq_pools)
    return RaagClassifier(trunk, trunk.out_dim, num_labels=num_labels,
                          tonic_mode=tonic_mode, aux_occupancy=aux_occupancy,
                          head_hidden=head_hidden, dropout=head_dropout,
                          side_dim=side_dim, side_out=side_out)


def param_groups(model, lr=1e-3, head_lr=None, weight_decay=1e-4):
    """One group: this backbone is trained from scratch, so there is nothing to protect."""
    return [{"params": [p for p in model.parameters() if p.requires_grad],
             "lr": lr, "weight_decay": weight_decay}]
