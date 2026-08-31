import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path

from .base import BaseSeqEmbedder

VINET_DIR = Path(__file__).resolve().parent.parent.parent / "Discogs-VINet"
CKPT_PATH = VINET_DIR / "logs/checkpoints/Discogs-VINet/model_checkpoint.pth"

# CQT params (must match what the model was trained with)
SR = 22050
HOP_LENGTH = 512        # ~43 fps
N_BINS = 84             # 7 octaves × 12 semitones
BINS_PER_OCTAVE = 12

# Sequence extraction: 5-second windows at 43 fps, 2.5-second stride.
# Minimum viable window is ~175 frames before the last conv collapses T to 0.
WINDOW_FRAMES = 215   # ~5 s
STRIDE_FRAMES = 108   # ~2.5 s


def _build_model():
    """Rebuild CQTNet matching the checkpoint's key names (features.* / proj.0.*)."""
    ch_in = 32
    bn = nn.BatchNorm2d
    features = nn.Sequential(
        nn.Conv2d(1, ch_in, (12, 3), dilation=(1, 1), padding=(6, 0), bias=False),
        bn(ch_in), nn.ReLU(inplace=True),
        nn.Conv2d(ch_in, 2 * ch_in, (13, 3), dilation=(1, 2), bias=False),
        bn(2 * ch_in), nn.ReLU(inplace=True),
        nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
        nn.Conv2d(2 * ch_in, 2 * ch_in, (13, 3), dilation=(1, 1), bias=False),
        bn(2 * ch_in), nn.ReLU(inplace=True),
        nn.Conv2d(2 * ch_in, 2 * ch_in, (3, 3), dilation=(1, 2), bias=False),
        bn(2 * ch_in), nn.ReLU(inplace=True),
        nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
        nn.Conv2d(2 * ch_in, 4 * ch_in, (3, 3), dilation=(1, 1), bias=False),
        bn(4 * ch_in), nn.ReLU(inplace=True),
        nn.Conv2d(4 * ch_in, 4 * ch_in, (3, 3), dilation=(1, 2), bias=False),
        bn(4 * ch_in), nn.ReLU(inplace=True),
        nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
        nn.Conv2d(4 * ch_in, 8 * ch_in, (3, 3), dilation=(1, 1), bias=False),
        bn(8 * ch_in), nn.ReLU(inplace=True),
        nn.Conv2d(8 * ch_in, 8 * ch_in, (3, 3), dilation=(1, 2), bias=False),
        bn(8 * ch_in), nn.ReLU(inplace=True),
        nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
        nn.Conv2d(8 * ch_in, 16 * ch_in, (3, 3), dilation=(1, 1), bias=False),
        nn.BatchNorm2d(16 * ch_in), nn.ReLU(inplace=True),
        nn.Conv2d(16 * ch_in, 16 * ch_in, (3, 3), dilation=(1, 2), bias=False),
        nn.BatchNorm2d(16 * ch_in), nn.ReLU(inplace=True),
    )

    class CQTNetCompat(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            # proj.0.weight matches nn.Sequential([nn.Linear(...)]) checkpoint keys
            self.proj = nn.Sequential(nn.Linear(16 * ch_in, 512, bias=False))

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.proj(x)
            return F.normalize(x, dim=-1)

    return CQTNetCompat()


class VINetEmbedder(BaseSeqEmbedder):
    """Discogs-VINet (ISMIR 2024) sequence embedder.

    Computes CQT (84 bins, sr=22050, hop=512) then slides 2-second windows
    with 1-second stride, producing (N_windows, 512) per clip.
    Uses the bundled pretrained checkpoint trained on Discogs version identification.
    """

    name = "vinet"

    def load(self):
        assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
        model = _build_model()
        raw_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        model.load_state_dict(raw_sd)
        model.eval()
        self._model = model

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        if sr != SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SR)

        cqt = librosa.cqt(
            audio_array,
            sr=SR,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )  # (F, T)
        cqt = np.abs(cqt).astype(np.float32)
        cqt /= np.max(cqt) + 1e-6  # normalize to [0,1]

        T = cqt.shape[1]
        if T < WINDOW_FRAMES:
            cqt = np.pad(cqt, ((0, 0), (0, WINDOW_FRAMES - T)))
            T = WINDOW_FRAMES

        # Build sliding windows: (N, 1, F, WINDOW_FRAMES)
        starts = list(range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES))
        chunks = np.stack([cqt[:, s : s + WINDOW_FRAMES] for s in starts], axis=0)
        chunks = chunks[:, np.newaxis, :, :]  # (N, 1, F, W)
        x = torch.from_numpy(chunks)

        with torch.no_grad():
            emb = self._model(x)  # (N, 512)

        return emb.numpy().astype(np.float32)
