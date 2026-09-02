"""Audio: decode, cache, tonic-normalise, separate, and CQT.

Every model in this folder reads its input through `clip_tensor()`. That is deliberate --
the three architectures differ in sample rate and in whether they want a waveform or a
spectrogram, but they must agree exactly on *which audio* they are looking at, or the
tonic and separation ablations compare two things at once.

## The three ways audio can be prepared

    raw                     the mixture, as decoded. The control.
    tonic-normalised        resampled by `tonic.shift_ratio` so Sa lands on a fixed
                            reference. The model no longer has to learn transposition.
    separated               the melody stem from ../source-separation (HPSS by default).

They compose: `clip_tensor(clip, sr, tonic="normalise", separate="hpss")`.

## The length policy, and the confound hiding in it

Resampling shifts pitch *and* duration -- a clip shifted down by 6 semitones comes out
1.41x longer. So a fixed-length input means the amount of *music* the model sees depends on
the clip's tonic, which is a leak: the network could read the tonic off the amount of
zero-padding rather than off the audio.

    length_policy="fixed"     every input is `seconds` long. Normalised clips are
                              zero-padded (shift up) or centre-cropped (shift down).
                              Simple, uses all 20 s, but carries the confound above.
    length_policy="musical"   every input holds the same amount of *musical time*
                              (`seconds / 2**0.5`, i.e. 14.14 s of a 20 s clip) in the same
                              number of samples, for every condition including the raw
                              control. Confound-free, 29 % of the audio unused, ~30 % faster.

Run the tonic experiment under "fixed" for the headline and re-run the winner under
"musical" to confirm the gain is not the padding. Both are cheap to ask for.
"""

import os
import warnings
from pathlib import Path

import numpy as np

from . import tonic as tonic_mod
from .paths import CACHE, SEP_DIR, add_sibling_paths

SR_HUBERT = 16000     # distilHuBERT's native rate
SR_JEEVSTER = 8000    # the jeevster ResNet was trained at 8 kHz
SR_CQT = 22050        # enough headroom for a 4-octave CQT off a ~55 Hz fmin

#: One cache, at the highest rate anything needs. The three models resample down from it on
#: the fly (~4 ms per 20 s clip with soxr, against ~200 ms to decode the mp3 again), which
#: keeps one copy of the corpus on disk instead of three. Disk here is the scarce resource.
SR_CACHE = SR_CQT

DEFAULT_SECONDS = 20.0   # the dataset's clip length
MUSICAL_FRACTION = 2.0 ** -0.5   # 1 / max shift ratio -- see the docstring


# ------------------------------------------------------------------ decode + cache


def _atomic_save(path, arr):
    """Write via a temp file and rename. np.save is not atomic, so a run killed mid-write
    otherwise leaves a truncated .npy that the *resumable* rerun happily skips -- the one
    failure mode that silently corrupts a cache instead of loudly crashing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _cache_path(kind, clip, sep=None, tag=None):
    parts = [kind, sep or "raw"]
    if tag:
        parts.append(tag)
    return CACHE / "/".join(parts) / (clip.clip_id.replace(".mp3", ".npy"))


def decode(clip, sr):
    """mp3 -> mono float32 at `sr`. No caching; use `cached_waveform` instead."""
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(clip.path), sr=sr, mono=True)
    return y.astype(np.float32)


def _separate(y, sr, backend):
    """The melody stem from ../source-separation. `backend=None` is a pass-through."""
    if not backend or backend == "none":
        return y
    add_sibling_paths()
    from separation import separate as _sep

    return np.asarray(_sep(y, sr, backend=backend).melody, dtype=np.float32)


def cached_waveform(clip, separate=None, build=True):
    """Decoded (and optionally separated) audio at SR_CACHE, memoised to disk as int16.

    int16 rather than float32 halves the cache and loses nothing: mp3 decodes to 16-bit
    resolution anyway. Peak-normalised, so loudness is not a feature the models can cheat
    with -- recording level correlates with the source video, and the source video
    correlates with the label.
    """
    p = _cache_path("audio", clip, sep=separate)
    if p.exists():
        return np.load(p).astype(np.float32) / 32767.0
    if not build:
        raise FileNotFoundError(f"{p} not cached; run scripts/00_build_cache.py")
    y = _separate(decode(clip, SR_CACHE), SR_CACHE, separate)
    peak = float(np.max(np.abs(y))) or 1.0
    _atomic_save(p, np.clip(y / peak * 32767.0, -32767, 32767).astype(np.int16))
    return (y / peak).astype(np.float32)


# ------------------------------------------------------------------ tonic normalisation


def resample_shift(y, sr_in, sr_out, ratio=1.0):
    """Resample `y` from `sr_in` to `sr_out` and multiply every frequency by `ratio`, in one
    pass.

    The pitch shift is a resample: claiming the audio was recorded at `sr_in * ratio` and
    converting it to `sr_out` plays it `ratio` times faster, which is `ratio` times higher.
    Exact in pitch and free of phase-vocoder artefacts, at the cost of changing duration by
    1/ratio -- see the length-policy discussion in the module docstring. The alternative
    (`librosa.effects.pitch_shift`) preserves duration but smears transients, and in this
    repertoire the continuous movement between swars *is* the signal (see the
    source-separation post-mortem in ../motif-classifier/plan.md), so smearing is the more
    expensive error.
    """
    import librosa

    if abs(ratio - 1.0) < 1e-9 and sr_in == sr_out:
        return y.astype(np.float32)
    return librosa.resample(y, orig_sr=sr_in * ratio, target_sr=sr_out,
                            res_type="soxr_hq").astype(np.float32)


def fit_length(y, n, where="centre"):
    """Zero-pad or crop to exactly `n` samples."""
    if len(y) == n:
        return y
    if len(y) < n:
        pad = n - len(y)
        if where == "centre":
            return np.pad(y, (pad // 2, pad - pad // 2))
        return np.pad(y, (0, pad))
    start = (len(y) - n) // 2 if where == "centre" else 0
    return y[start:start + n]


def clip_tensor(clip, sr, tonic="none", separate=None, seconds=DEFAULT_SECONDS,
                length_policy="fixed", tonic_hz=None, build=True):
    """The one entry point every model uses. Returns float32, exactly `n_samples` long.

    tonic : "none"       leave the audio alone (the model may still be *told* the tonic)
            "normalise"  resample so Sa lands on tonic.REF_TONIC_HZ
    """
    hz = clip.tonic_hz if tonic_hz is None else tonic_hz
    n_out = n_samples(sr, seconds, length_policy)
    if tonic == "none":
        ratio = 1.0
    elif tonic == "normalise":
        ratio = tonic_mod.shift_ratio(hz)
    else:
        raise ValueError(f"tonic must be 'none' or 'normalise', got {tonic!r}")

    y = cached_waveform(clip, separate=separate, build=build)
    if length_policy == "musical":
        # take exactly as much source as will resample to n_out samples, so the musical
        # content is identical across conditions and the padding confound disappears
        y = fit_length(y, min(int(round(n_out * ratio * SR_CACHE / sr)), len(y)))
    return fit_length(resample_shift(y, SR_CACHE, sr, ratio), n_out)


def n_samples(sr, seconds=DEFAULT_SECONDS, length_policy="fixed"):
    return int(round(sr * seconds * (MUSICAL_FRACTION if length_policy == "musical" else 1.0)))


# ------------------------------------------------------------------ CQT


CQT_BINS_PER_OCTAVE = 36    # 33.3 cents/bin -- fine enough to see meend, coarse enough to batch
CQT_OCTAVES = 4             # mandra Sa .. ati-taar Sa
CQT_HOP = 1024              # 21.5 frames/s at 22050 -> 430 frames for 20 s


def cqt(clip, tonic="anchor", separate=None, seconds=DEFAULT_SECONDS,
        bins_per_octave=CQT_BINS_PER_OCTAVE, octaves=CQT_OCTAVES, hop=CQT_HOP,
        fmin_hz=55.0, tonic_hz=None, build=True):
    """log-magnitude CQT, (n_bins, n_frames) float32.

    tonic : "anchor"   `fmin` = Sa, one octave below the canonical tonic. **Bin 0 is Sa
                       exactly** -- no rolling, no rounding to the bin grid, and the
                       representation is tonic-invariant by construction. This is the whole
                       argument for the C architecture.
            "none"     `fmin` = a fixed `fmin_hz` for every clip: absolute pitch. The
                       control that says how much of C's performance is the anchoring.
    """
    import librosa

    hz = clip.tonic_hz if tonic_hz is None else tonic_hz
    fmin = tonic_mod.anchor_fmin(hz) if tonic == "anchor" else float(fmin_hz)
    y = clip_tensor(clip, SR_CQT, tonic="none", separate=separate, seconds=seconds,
                    build=build)
    C = np.abs(librosa.cqt(y, sr=SR_CQT, fmin=fmin, n_bins=bins_per_octave * octaves,
                           bins_per_octave=bins_per_octave, hop_length=hop))
    return librosa.amplitude_to_db(C, ref=np.max).astype(np.float32)


def cached_cqt(clip, tonic="anchor", separate=None, build=True, **kw):
    """`cqt` memoised as float16. ~150 kB/clip, so the whole corpus is ~300 MB.

    The cache key includes the anchor frequency, not just `tonic="anchor"`. It did not
    until 2026-08-31, and the omission silently voided the `c2_shuffled` control: the
    shuffled run asked for a CQT anchored at a deliberately wrong Sa, hit the cache entry
    written by the correctly-anchored run, and trained on identical data. It scored 0.313
    against c2's 0.302 -- which read as "the tonic does not matter" when it actually meant
    "the tonic never reached the model". Any cache key that omits an input that changes
    the output is this bug waiting to happen.
    """
    bpo = kw.get("bins_per_octave", CQT_BINS_PER_OCTAVE)
    oct_ = kw.get("octaves", CQT_OCTAVES)
    tag = f"{tonic}_{bpo}x{oct_}"
    if tonic == "anchor":
        hz = kw.get("tonic_hz") if kw.get("tonic_hz") is not None else clip.tonic_hz
        tag += f"_fmin{tonic_mod.anchor_fmin(hz):.3f}"
    else:
        tag += f"_fmin{float(kw.get('fmin_hz', 55.0)):.3f}"
    p = _cache_path("cqt", clip, sep=separate, tag=tag)
    if p.exists():
        return np.load(p).astype(np.float32)
    if not build:
        raise FileNotFoundError(f"{p} not cached; run scripts/00_build_cache.py")
    C = cqt(clip, tonic=tonic, separate=separate, **kw)
    _atomic_save(p, C.astype(np.float16))
    return C


def fold_octaves(C, bins_per_octave=CQT_BINS_PER_OCTAVE):
    """(n_bins, n_frames) -> (bins_per_octave, n_frames), summing over octaves.

    Octave equivalence is not an approximation in this repertoire: a raag's identity is its
    swar set and the movement between them, not the register they are sung in, and pitch
    trackers make octave errors often enough that ../motif-classifier drops saptak marks
    entirely. Folding a Sa-anchored CQT gives a 36-bin (33 cents) profile whose bin 0 is Sa.
    """
    n_bins = C.shape[0]
    if n_bins % bins_per_octave:
        raise ValueError(f"{n_bins} bins is not a whole number of {bins_per_octave}-bin octaves")
    return C.reshape(n_bins // bins_per_octave, bins_per_octave, -1).sum(axis=0)


def db_to_amplitude(C_db):
    """Undo `amplitude_to_db(ref=max)`. The CQT is cached in dB because that is what a CNN
    wants to see; a *histogram* wants energy, where a 40 dB-down bin is negligible rather
    than half-scale."""
    return np.power(10.0, np.asarray(C_db, dtype=np.float32) / 20.0)
