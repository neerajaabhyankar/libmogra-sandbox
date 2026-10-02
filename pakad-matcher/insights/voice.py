"""Voice-only loudness: the clip's spectrum minus the drone's steady part.

The tanpura (and any steady drone) sounds the same partials all the way through, so for each
frequency bin its level is roughly the bin's quiet-end level over the clip (a low percentile).
What rises above that is the voice (or another moving instrument). Summed over the voice band, it
gives a loudness that falls during a breath even when the tanpura keeps the overall level up and
the pitch tracker locks onto Sa. (Memory: drone vs voice is unsolved in general -- this is only
validated as a pause cue, against Neeraja's nyas windows, in insights/evaluate.py.)
"""

import numpy as np

import config as C

V = C.INSIGHT_VOICE


def voice_db(wav, hop, n):
    """dB per contour frame of the energy above the drone floor, in the voice band."""
    import librosa
    y, sr = librosa.load(wav, sr=V["sr"], mono=True)
    h = max(1, int(round(hop * sr)))
    S = np.abs(librosa.stft(y, n_fft=V["n_fft"], hop_length=h)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=V["n_fft"])
    band = (f >= V["band_hz"][0]) & (f <= V["band_hz"][1])
    floor = np.percentile(S[band], V["floor_pct"], axis=1, keepdims=True)
    resid = np.maximum(S[band] - V["floor_mult"] * floor, 0.0).sum(0)
    db = 10 * np.log10(resid + 1e-10)
    db = np.maximum(db, np.percentile(db, 95) - V["range_db"])     # clip the bottom
    return np.pad(db, (0, max(0, n - len(db))), mode="edge")[:n]
