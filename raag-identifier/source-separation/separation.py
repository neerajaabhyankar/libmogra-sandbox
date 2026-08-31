"""Pull the melody out of a Hindustani recording, cheaply.

The goal here is deliberately modest. We do not need clean stems; we need the pitch
tracker downstream to lock onto the *main* melodic line — voice, or sarangi/sitar/flute,
or several of them in a jugalbandi — instead of being dragged around by tabla transients
and the tanpura drone. Some bleed is fine.

Backends, cheapest first. All share one interface, so callers pick by name and never
import a model directly:

    hpss        librosa harmonic/percussive median filtering. Free, no new dependency,
                no model download, ~realtime. Removes tabla; keeps the drone.
    hpss+drone  the same, then a drone notch (see `suppress_drone`). Removes tabla and
                most of the tanpura.
    demucs      HT-Demucs `htdemucs`, the 4-stem Western model (vocals/drums/bass/other).
                Much better at voice, but it was trained on Western pop: tabla strokes are
                strongly pitched and leak into the vocal stem, and there is no tanpura in
                its vocabulary. Needs `pip install demucs` and a model download.
    none        pass the mixture through unchanged — the control condition.

What is deliberately NOT here: BS-RoFormer / Mel-Band RoFormer fine-tuned on Saraga
multitracks, which is the actually-good answer and a research project of its own. The
backend registry is the seam where it would slot in.

    from separation import separate, BACKENDS
    stems = separate(audio, sr, backend="hpss")
    stems.melody   # np.ndarray, same sr
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


@dataclass
class Stems:
    """What a backend returns. `melody` is the only field callers must use."""

    melody: np.ndarray
    sr: int
    backend: str
    percussion: np.ndarray = None
    drone: np.ndarray = None
    residual: np.ndarray = None
    meta: dict = field(default_factory=dict)

    def energy_split(self):
        """Fraction of the input's energy in each stem — the cheapest sanity check.

        A melody stem holding ~all the energy means the backend did nothing; one holding
        almost none means it ate the music.
        """
        def e(x):
            return float(np.sum(np.square(x))) if x is not None and len(x) else 0.0
        parts = {"melody": e(self.melody), "percussion": e(self.percussion),
                 "drone": e(self.drone), "residual": e(self.residual)}
        tot = sum(parts.values()) or 1.0
        return {k: v / tot for k, v in parts.items()}


# ---------------------------------------------------------------- backends


def _passthrough(audio, sr, **kw):
    return Stems(melody=audio, sr=sr, backend="none")


def _hpss(audio, sr, margin=3.0, n_fft=2048, **kw):
    """Harmonic/percussive separation by median filtering the spectrogram.

    Tabla is broadband and vertical in the spectrogram; sung/bowed melody is narrowband
    and horizontal. Median-filtering along each axis separates them with no model at all.
    `margin > 1` makes the split conservative — energy that is not clearly one or the
    other is dropped from both, which is what we want, since we would rather lose a little
    melody than keep a tabla stroke that the pitch tracker will chase.
    """
    import librosa

    S = librosa.stft(audio, n_fft=n_fft)
    H, P = librosa.decompose.hpss(S, margin=margin)
    n = len(audio)
    mel = librosa.istft(H, length=n)
    perc = librosa.istft(P, length=n)
    return Stems(melody=mel, sr=sr, backend="hpss", percussion=perc,
                 residual=audio - mel - perc, meta={"margin": margin})


def suppress_drone(audio, sr, n_fft=2048, quantile=0.7, width_bins=2, floor_db=-18.0):
    """Notch out whatever is *always there* — the tanpura.

    A tanpura holds the same handful of pitches for the entire recording, so its bins have
    a high floor across time, while a melodic note occupies a bin only in bursts. Taking a
    per-bin low quantile over time estimates that floor, and subtracting a bounded amount
    of it removes the drone while leaving melody largely intact.

    This is a blunt instrument: it also attenuates a genuinely sustained nyas note on the
    same pitch as the drone (which, for Sa, is exactly the interesting case). `floor_db`
    bounds how much any bin can be cut so that a long Sa is dimmed, not deleted.
    """
    import librosa

    S = librosa.stft(audio, n_fft=n_fft)
    mag, phase = np.abs(S), np.angle(S)
    floor = np.quantile(mag, quantile, axis=1, keepdims=True)
    if width_bins:  # the drone is not exactly one bin wide
        k = 2 * width_bins + 1
        floor = np.array([
            np.max(floor[max(i - width_bins, 0): i + width_bins + 1, 0])
            for i in range(floor.shape[0])
        ]).reshape(-1, 1)
    reduced = np.maximum(mag - floor, mag * (10.0 ** (floor_db / 20.0)))
    out = librosa.istft(reduced * np.exp(1j * phase), length=len(audio))
    return out, librosa.istft((mag - reduced) * np.exp(1j * phase), length=len(audio))


def _hpss_drone(audio, sr, margin=3.0, n_fft=2048, **kw):
    s = _hpss(audio, sr, margin=margin, n_fft=n_fft)
    mel, drone = suppress_drone(s.melody, sr, n_fft=n_fft, **{
        k: v for k, v in kw.items() if k in ("quantile", "width_bins", "floor_db")})
    return Stems(melody=mel, sr=sr, backend="hpss+drone", percussion=s.percussion,
                 drone=drone, residual=s.residual, meta=s.meta)


@lru_cache(maxsize=1)
def _demucs_model(name="htdemucs"):
    from demucs.pretrained import get_model

    m = get_model(name)
    m.eval()
    return m


def _demucs(audio, sr, name="htdemucs", device=None, **kw):
    """HT-Demucs. `vocals` is the melody stem; `drums` is the closest thing to tabla.

    Caveat worth keeping in view: this model has never heard a tanpura or a tabla. Pitched
    tabla strokes leak into `vocals`, and the drone lands wherever it lands. It is here
    because it is a strong, mature, one-line baseline — not because it is right for this
    repertoire.
    """
    import torch
    from demucs.apply import apply_model

    model = _demucs_model(name)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    import librosa

    wav = audio
    if sr != model.samplerate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=model.samplerate)
    x = torch.from_numpy(np.ascontiguousarray(wav)).float()
    x = x.unsqueeze(0).repeat(model.audio_channels, 1).unsqueeze(0)  # (1, ch, T)
    ref = x.mean(dim=(0, 1))
    x = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, x, device=device, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    stems = {n: out[i].mean(dim=0).cpu().numpy() for i, n in enumerate(model.sources)}
    if sr != model.samplerate:
        stems = {k: librosa.resample(v, orig_sr=model.samplerate, target_sr=sr)
                 for k, v in stems.items()}
    n = len(audio)
    fix = lambda v: np.pad(v, (0, max(0, n - len(v))))[:n]
    return Stems(
        melody=fix(stems.get("vocals")), sr=sr, backend=f"demucs:{name}",
        percussion=fix(stems.get("drums")) if "drums" in stems else None,
        residual=fix(stems.get("other")) if "other" in stems else None,
        drone=fix(stems.get("bass")) if "bass" in stems else None,
        meta={"model": name, "device": device, "sources": list(model.sources)},
    )


def _demucs_drone(audio, sr, **kw):
    s = _demucs(audio, sr, **kw)
    mel, drone = suppress_drone(s.melody, sr)
    return Stems(melody=mel, sr=sr, backend=s.backend + "+drone", percussion=s.percussion,
                 drone=drone, residual=s.residual, meta=s.meta)


BACKENDS = {
    "none": _passthrough,
    "hpss": _hpss,
    "hpss+drone": _hpss_drone,
    "demucs": _demucs,
    "demucs+drone": _demucs_drone,
}


def separate(audio, sr, backend="hpss", **kw) -> Stems:
    """Melody-first separation. `backend` is a key of BACKENDS."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; have {sorted(BACKENDS)}")
    audio = np.asarray(audio, dtype=np.float32).ravel()
    return BACKENDS[backend](audio, sr, **kw)


def available():
    """Which backends can actually run here, so callers can degrade instead of crash."""
    out = {}
    for name in BACKENDS:
        if name.startswith("demucs"):
            try:
                import demucs  # noqa: F401

                out[name] = True
            except ImportError:
                out[name] = False
        else:
            out[name] = True
    return out
