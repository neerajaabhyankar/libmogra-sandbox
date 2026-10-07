"""Per-frame feature blocks for the adaptation head (adapt.py), all on one 10 ms grid.

    extract(name, path, t0, t1)   -> (frames, d) block of audio `path` from t0 to t1 s, cached
    track_block(f0_hz, conf, hop, n) -> a pitch track (e.g. Melodia's) as a block
    BLOCKS                         the audio-based blocks: cqt, bp (Basic Pitch's three maps)

A pitch track becomes a block by a bump at its pitch over the head's own pitch bins (so the head
can read it positionally), plus its confidence and a voiced flag. Callers bring tracks they
already have (pakad-matcher brings Melodia); nothing here imports them.
"""

import hashlib

import librosa
import numpy as np

from . import config as C

HOP = C.ADAPT["hop_s"]
B = C.ADAPT


def bins_hz():
    """Centre of every pitch bin of the head (and of track blocks)."""
    return B["fmin_hz"] * 2.0 ** (np.arange(B["n_bins"]) * B["cents_per_bin"] / 1200.0)


def on_grid(x, hop, n):
    """Frames of `x` (frames, ...) at `hop` s, picked at the n frames of the 10 ms grid."""
    idx = np.minimum((np.arange(n) * HOP / hop).round().astype(int), len(x) - 1)
    return x[idx]


def _cqt(audio, sr):
    c = librosa.cqt(audio, sr=sr, hop_length=int(HOP * sr), fmin=B["fmin_hz"],
                    n_bins=B["cqt_bins"], bins_per_octave=36)
    return librosa.amplitude_to_db(np.abs(c), ref=np.max).T / 80.0 + 1.0, HOP


def _bp(audio, sr):
    from .models.basic_pitch import adapter
    m = adapter.maps(audio)
    return np.concatenate([m["contour"], m["note"], m["onset"]], axis=1), 256 / sr


BLOCKS = {"cqt": (16000, _cqt), "bp": (22050, _bp)}


def extract(name, path, t0, t1):
    """Block `name` for `path` from t0 to t1 s on the 10 ms grid; cached under cache/features/."""
    key = hashlib.md5(str(path).encode()).hexdigest()[:12]
    f = C.CACHE_DIR / "features" / name / f"{key}_{t0:.2f}_{t1:.2f}.npy"
    if f.exists():
        return np.load(f)
    sr, fn = BLOCKS[name]
    audio, _ = librosa.load(path, sr=sr, mono=True, offset=t0, duration=t1 - t0)
    x, hop = fn(audio, sr)
    x = on_grid(x, hop, int(round((t1 - t0) / HOP))).astype(np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, x)
    return x


def track_block(f0_hz, conf, hop, n):
    """A pitch track as a block on the grid: bump at its pitch (1 bin wide), confidence, voiced."""
    f0, cf = on_grid(np.asarray(f0_hz, float), hop, n), on_grid(np.asarray(conf, float), hop, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = 1200 * np.log2(f0 / B["fmin_hz"]) / B["cents_per_bin"]
    bump = np.exp(-0.5 * (np.arange(B["n_bins"])[None] - b[:, None]) ** 2)
    bump[~(f0 > 0)] = 0.0
    return np.concatenate([bump, cf[:, None], (f0 > 0)[:, None]], axis=1).astype(np.float32)
