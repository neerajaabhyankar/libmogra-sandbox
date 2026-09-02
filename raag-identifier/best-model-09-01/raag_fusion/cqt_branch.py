"""Branch 1 -- a small 2-D ResNet over a Sa-anchored CQT, with a swar-template head.

**The representation.** A constant-Q transform whose `fmin` is the clip's own Sa. Bin 0 is
Sa exactly, bin 3 is one semitone above it, and that mapping is identical for every
recording in the world. Nothing is resampled, nothing is interpolated, and the network
never has to learn transposition from 1810 examples.

**The trunk** pools time hard and frequency gently, because the frequency axis *is* the
label -- which swars are present, and where the energy sits between them. What comes out is
a (channels x frequency) profile rather than a single pooled vector.

**The head** does not classify with 50 free weight vectors. It predicts one 12-bin swar
profile and compares it to 50 per-raag templates by chi-square. The templates start at the
libmogra database's swar occupancy and are then learned, so a raag with 18 training clips
inherits a usable template instead of starting from noise. (The database values live in the
checkpoint, so nothing here needs libmogra installed.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BINS_PER_OCTAVE = 36     # 33.3 cents -- fine enough to see meend between swars
OCTAVES = 4              # mandra Sa .. ati-taar Sa
N_BINS = BINS_PER_OCTAVE * OCTAVES
HOP = 1024               # 21.5 frames/s at 22050 Hz
N_FRAMES = 431           # what 20 s comes to, and what the network was trained on
EPS = 1e-8


def features(y22050, tonic_hz):
    """20 s of peak-normalised 22.05 kHz audio -> (1, 144, 431) float32, ready for `Net`."""
    import librosa

    from .tonic import anchor_fmin

    C = np.abs(librosa.cqt(y22050, sr=22050, fmin=anchor_fmin(tonic_hz), n_bins=N_BINS,
                           bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP))
    C = librosa.amplitude_to_db(C, ref=np.max).astype(np.float16).astype(np.float32)
    if C.shape[1] >= N_FRAMES:
        C = C[:, :N_FRAMES]
    else:
        C = np.pad(C, ((0, 0), (0, N_FRAMES - C.shape[1])), constant_values=C.min())
    return ((C + 80.0) / 80.0)[None]          # dB in [-80, 0] -> roughly [0, 1]


class _Block(nn.Module):
    """Pre-activation residual block with a configurable (frequency, time) pool."""

    def __init__(self, cin, cout, pool=(2, 2), dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.pool = pool

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.drop(F.relu(h + self.skip(x)))
        return F.max_pool2d(h, self.pool)


class Backbone(nn.Module):
    """(B, 1, 144, 431) -> (B, 432): 24 channels x 18 surviving frequency bins."""

    def __init__(self, channels=(32, 64, 96, 128), proj_channels=24, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels[0], 5, padding=2, bias=False),
                                  nn.BatchNorm2d(channels[0]), nn.ReLU())
        pools = [(2, 2), (2, 2), (2, 2), (1, 2)]      # 144 frequency bins -> 18
        blocks, cin = [], channels[0]
        for cout, pool in zip(channels, pools):
            blocks.append(_Block(cin, cout, pool=pool, dropout=dropout))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(cin, proj_channels, 1, bias=False)
        self.out_dim = proj_channels * (N_BINS // 8)

    def forward(self, x):
        h = self.proj(self.blocks(self.stem(x)))
        return h.mean(dim=-1).flatten(1)              # pool time only, keep frequency


class TemplateHead(nn.Module):
    """(B, 432) -> (B, 50), through a 12-bin swar profile scored against 50 templates.

    `lam` is how far the templates are pulled back toward the database (0.3 here): the
    reference each raag is scored against is `0.7 * learned + 0.3 * database`.
    """

    def __init__(self, feature_dim, n_raags=50, n_bins=12, lam=0.3, hidden=128, dropout=0.2):
        super().__init__()
        self.lam = float(lam)
        self.to_profile = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_bins))
        # filled from the checkpoint; `train.py` initialises them from libmogra instead
        self.register_buffer("db_templates", torch.zeros(n_raags, n_bins))
        self.learned_logits = nn.Parameter(torch.zeros(n_raags, n_bins))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(n_raags))

    def templates(self):
        learned = F.softmax(self.learned_logits, dim=-1)
        return (1.0 - self.lam) * learned + self.lam * self.db_templates

    def forward(self, h):
        p = F.softmax(self.to_profile(h), dim=-1)
        refs = self.templates()
        chi2 = 0.5 * (((p[:, None, :] - refs[None]) ** 2)
                      / (p[:, None, :] + refs[None] + EPS)).sum(-1)
        return -self.scale.abs() * chi2 + self.bias


class Net(nn.Module):
    """The whole branch: Sa-anchored CQT in, 50 logits out."""

    def __init__(self, n_raags=50, lam=0.3):
        super().__init__()
        self.backbone = Backbone()
        self.head = TemplateHead(self.backbone.out_dim, n_raags=n_raags, lam=lam)

    def forward(self, x):
        return self.head(self.backbone(x))
