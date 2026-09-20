"""Branch 2 -- a pitch histogram and a linear model. Deliberately naive.

CREPE gives a frame-level f0 track; every voiced frame is expressed in cents above Sa,
folded into one octave, and dropped into 120 bins. The histogram is blurred slightly (so
two performances tuned a few cents apart still overlap) and raised to the power 0.5 (so one
long held nyas note cannot swamp every other swar). A multinomial logistic regression then
reads the 120 numbers.

There is no note segmentation, no phrase model, no grammar -- and on the held-out test
split this scored 0.373, exactly matching the elaborate symbolic pipeline it was meant to
be a baseline for. That is why it, and not the elaborate one, is the second branch here: it
is as accurate, and it needs one pip package instead of a native Vamp plugin.

It is also *wrong in different places* from the CQT branch, which is the entire point --
the two agree on only 29 % of test clips, so averaging them beats both.
"""

import numpy as np

from . import pitch

# CREPE's own settings live in `pitch`; re-exported here so existing callers and the
# README keep working.
SR = pitch.SR
HOP = pitch.HOP
CONFIDENCE = pitch.CONFIDENCE
MODEL_SIZE = pitch.MODEL_SIZE

N_BINS = 120              # 10 cents per bin
SMOOTH = 1.0
POWER = 0.5               # applied by `features`, not by `histogram` -- see below


def f0_track(y16000, device="cpu", dither_seed=0):
    """(f0 in Hz, voiced mask), one value per 10 ms frame.

    Kept as the pair this has always returned. New code should call `pitch.track`, which
    returns a `PitchTrack` carrying the same two arrays plus what can be read off them.
    """
    t = pitch.track(y16000, device=device, dither_seed=dither_seed)
    return t.f0_hz, t.voiced


def histogram(f0_hz, voiced, tonic_hz, n_bins=N_BINS, smooth=SMOOTH):
    """Voiced frames -> a (n_bins,) octave-folded pitch histogram, summing to 1.

    **This is a distribution over time, not a feature vector.** Each bin is the share of
    the voiced passage spent at that pitch class, so a swar sung four times as long as
    another is four times as tall, and the array can be drawn as "share of your time" with
    nothing further done to it. `features` is what turns it into the classifier's input.
    """
    f0 = np.asarray(f0_hz, dtype=float)[np.asarray(voiced, dtype=bool)]
    cents = 1200.0 * np.log2(np.clip(f0, 1e-6, None) / float(tonic_hz))
    cents = cents[np.isfinite(cents)]
    if cents.size < 5:
        return np.zeros(n_bins)
    idx = np.floor((cents % 1200.0) * (n_bins / 1200.0)).astype(int) % n_bins
    H = np.zeros(n_bins)
    np.add.at(H, idx, 1.0)
    if smooth > 0:                       # circular Gaussian blur, done by FFT
        d = np.arange(n_bins)
        d = np.minimum(d, n_bins - d)
        kern = np.exp(-0.5 * (d / smooth) ** 2)
        H = np.maximum(np.real(np.fft.ifft(np.fft.fft(H) * np.fft.fft(kern / kern.sum()))), 0.0)
    total = H.sum()
    return H / total if total > 0 else H


def features(hist, power=POWER):
    """A histogram -> what `LinearModel` was fitted on.

    The compression is the feature engineering, not a property of the histogram: raising
    to 0.5 stops one long held nyas note from swamping every other swar, which helps a
    linear model and would misreport the drawn histogram by a square root.

    Normalising after the power is what the model saw, and doing it in two steps changes
    nothing -- scaling a histogram by 1/S then raising to p divides every bin by S**p,
    which the renormalisation takes straight back out.
    """
    H = np.asarray(hist, dtype=float) ** power
    total = H.sum()
    return H / total if total > 0 else H


def profile(track, tonic_hz, n_bins=N_BINS, smooth=SMOOTH):
    """A `pitch.PitchTrack` -> its histogram. The shape everything new should use."""
    return histogram(track.f0_hz, track.voiced, tonic_hz, n_bins=n_bins, smooth=smooth)


class LinearModel:
    """Standardise, then multinomial logistic regression. Fitted by `train.py`.

    Stored as four arrays rather than a pickled scikit-learn object: a pickle ties the
    weights to the version of scikit-learn that made them, and this is four lines of numpy.
    """

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept = np.asarray(coef), np.asarray(intercept)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(z["mean"], z["scale"], z["coef"], z["intercept"])

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef,
                 intercept=self.intercept)

    def scores(self, hist):
        """(n_bins,) histogram -> (n_raags,) unnormalised scores."""
        z = (np.asarray(hist, dtype=float) - self.mean) / self.scale
        return self.coef @ z + self.intercept
