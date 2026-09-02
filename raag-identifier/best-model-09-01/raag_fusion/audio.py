"""Getting a waveform into the exact shape each branch was trained on.

The two branches disagree on purpose, and the disagreement is copied verbatim from the
training pipeline rather than tidied up:

    the CQT branch    22.05 kHz, peak-normalised, exactly 20 s per window
    the melody branch 16 kHz, *not* normalised -- torchcrepe saw the raw decode

Peak normalisation matters for the CQT branch because recording level correlates with the
source video and the source video correlates with the raag; normalising removes a cue the
network could otherwise cheat with. It was never applied on the melody side, and matching
training beats being consistent.
"""

import numpy as np

SR_CQT = 22050
SR_CREPE = 16000
WINDOW_SECONDS = 20.0        # the clip length every training example had


def load(path, sr=None):
    """Any audio file -> (mono float32, sample_rate). Uses librosa, so anything ffmpeg or
    soundfile can open works."""
    import librosa

    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), int(sr_out)


def resample(y, sr_in, sr_out):
    import librosa

    if sr_in == sr_out:
        return np.asarray(y, dtype=np.float32)
    return librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr_in,
                            target_sr=sr_out, res_type="soxr_hq").astype(np.float32)


def peak_normalise(y):
    peak = float(np.max(np.abs(y))) or 1.0
    return (np.asarray(y, dtype=np.float32) / peak).astype(np.float32)


def fit_length(y, n):
    """Centre-crop or centre-pad to exactly `n` samples, as the training loader did."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == n:
        return y
    if len(y) < n:
        pad = n - len(y)
        return np.pad(y, (pad // 2, pad - pad // 2))
    start = (len(y) - n) // 2
    return y[start:start + n]


def windows(y, sr, seconds=WINDOW_SECONDS):
    """Split a recording into `seconds`-long windows, **centred** on it.

    Every training example was one 20 s clip, centre-cropped from a slightly longer one, so
    a 24 s recording must be scored on its middle 20 s and not its first 20 s. Tiling from
    the start instead was worth up to 9 points of probability mass on individual clips, and
    changed the top-1 answer on some of them.

    The number of windows is the recording length rounded to the nearest whole window, so
    47 s gives two windows of real audio rather than two plus a 7 s sliver, and 30 s gives
    two windows the second of which is half padding.
    """
    n = int(round(sr * seconds))
    k = max(1, int(len(y) / n + 0.5))
    start = max(0, (len(y) - k * n) // 2)
    return [fit_length(y[start + i * n:start + (i + 1) * n], n) for i in range(k)]
