"""Run a model over time ranges of an audio file, once, and keep the tracks.

    transcribe(model, key, path, segments)   -> transcribes whatever is not cached yet
    load(model, key)                         -> [(t0, Track)] for every cached range

`key` names the recording (pakad-matcher passes its video id); `segments` are (t0, t1) in seconds.
Audio files are only read, never written. Cache: config.CACHE_DIR/<model>/<key>.npz.
"""

import importlib

import librosa
import numpy as np

from . import config as C
from .contract import Track


def adapter(model):
    return importlib.import_module(f"transcriber.models.{model}.adapter")


def _path(model, key):
    return C.CACHE_DIR / model / f"{key}.npz"


def _read(model, key):
    p = _path(model, key)
    if not p.exists():
        return {}
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def load(model, key):
    z = _read(model, key)
    out = []
    for k in sorted(k for k in z if k.endswith("|f0")):
        seg = k[:-3]
        out.append((float(seg.split("-")[0]),
                    Track(z[k], float(z[f"{seg}|hop"]), z.get(f"{seg}|conf"))))
    return out


def transcribe(model, key, path, segments):
    """Transcribe the ranges of `path` not yet cached for `model`. Returns how many were new."""
    ad, z = adapter(model), _read(model, key)
    new = 0
    for t0, t1 in segments:
        seg = f"{t0:.2f}-{t1:.2f}"
        if f"{seg}|f0" in z:
            continue
        audio, sr = librosa.load(path, sr=ad.SR, mono=True, offset=t0, duration=t1 - t0)
        tr = ad.transcribe(audio, sr)
        z[f"{seg}|f0"], z[f"{seg}|hop"] = tr.f0_hz.astype(np.float32), np.float64(tr.hop_s)
        if tr.confidence is not None:
            z[f"{seg}|conf"] = tr.confidence.astype(np.float32)
        new += 1
    if new:
        _path(model, key).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_path(model, key), **z)
    return new
