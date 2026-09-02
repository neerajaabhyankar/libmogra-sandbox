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

SR = 16000
HOP = 160                 # 10 ms
CONFIDENCE = 0.4          # torchcrepe periodicity below this is treated as unvoiced
MODEL_SIZE = "tiny"
N_BINS = 120              # 10 cents per bin
SMOOTH = 1.0
POWER = 0.5


def f0_track(y16000, device="cpu", dither_seed=0):
    """(f0 in Hz, voiced mask), one value per 10 ms frame.

    **The dither seed is not optional decoration.** torchcrepe decodes pitch to a 20-cent
    bin grid and then adds triangular noise of +-20 cents to every frame to hide the
    quantisation (`torchcrepe.convert.dither`), drawn from numpy's *global* RNG. Left
    alone, that makes this function return a different answer every process: measured on
    one 24 s clip, the model's top-1 probability moved between 0.39 and 0.61 and the
    ranking below first place reshuffled. Seeding fixes the draw, so the same recording
    always gets the same answer.

    The caller's RNG state is saved and restored, because quietly reseeding numpy is not
    a reasonable side effect of asking for a pitch track.
    """
    import torch
    import torchcrepe

    wav = torch.from_numpy(np.ascontiguousarray(y16000)).float().unsqueeze(0)
    state = np.random.get_state()
    try:
        np.random.seed(dither_seed)
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                wav, SR, hop_length=HOP, fmin=50.0, fmax=2000.0, model=MODEL_SIZE,
                return_periodicity=True, batch_size=512, device=device,
                decoder=torchcrepe.decode.weighted_argmax)
    finally:
        np.random.set_state(state)
    return f0.squeeze(0).numpy(), periodicity.squeeze(0).numpy() >= CONFIDENCE


def histogram(f0_hz, voiced, tonic_hz, n_bins=N_BINS, smooth=SMOOTH, power=POWER):
    """Voiced frames -> a (n_bins,) octave-folded pitch histogram, summing to 1."""
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
    H = H ** power
    total = H.sum()
    return H / total if total > 0 else H


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
