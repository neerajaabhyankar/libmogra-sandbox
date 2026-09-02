"""Sa arithmetic, and getting Sa out of a hum.

Everything this model does is relative to the tonic. A raag is a set of intervals above
Sa, not a set of frequencies, so the same melody sung with Sa at 110 Hz and at 220 Hz is
the same raag -- and a model told the wrong Sa is not slightly wrong, it is answering a
different question. The survey behind this model measured the cost directly: the identical
network scored 0.302 with Sa anchored and 0.087 with the tonics permuted, against 0.020 for
guessing.

So the tonic is a required input, not an option. Two honest ways to supply it: type the
number, or hum a steady Sa for a few seconds and let `from_hum` read it.
"""

import numpy as np


def canonical(tonic_hz, lo_hz=110.0):
    """The tonic folded into [lo_hz, 2*lo_hz).

    A raag is octave-invariant, so a tonic of 110, 220 or 440 Hz should give the same
    representation. Folding makes that exactly true instead of approximately.
    """
    f = float(tonic_hz)
    while f < lo_hz:
        f *= 2.0
    while f >= 2.0 * lo_hz:
        f /= 2.0
    return f


def anchor_fmin(tonic_hz, octaves_below=1, lo_hz=110.0):
    """The CQT `fmin` that puts Sa on bin 0, one octave below the canonical Sa."""
    return canonical(tonic_hz, lo_hz) / (2.0 ** octaves_below)


def from_hum(audio, sr, fmin_hz=60.0, fmax_hz=600.0):
    """A few seconds of a held Sa -> its frequency in Hz.

    The median of the voiced pitch track, not the mean: a hum usually starts and ends with
    a scoop, and the median ignores both instead of averaging them into the answer.
    """
    import librosa

    f0, voiced, _prob = librosa.pyin(np.asarray(audio, dtype=float), sr=sr,
                                     fmin=fmin_hz, fmax=fmax_hz, fill_na=np.nan)
    f0 = f0[np.isfinite(f0) & np.asarray(voiced, dtype=bool)]
    if f0.size < 10:
        raise ValueError("could not hear a steady pitch -- hum one note, louder, for ~5 s")
    return float(np.median(f0))
