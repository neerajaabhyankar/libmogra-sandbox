"""M9 — time-delayed melody surfaces: the un-quantized representation, done properly.

Blurring the 12-swar histogram (M8) was the weak form of "stop quantizing", and it failed:
smearing a note across neighbouring swars destroys exactly the distinction that separates
raags differing by one komal/shuddha swap. The strong form does not quantize at all.

Following Gulati et al.'s time-delayed melody surface, a clip is represented by the 2D
histogram of

    ( pitch at time t , pitch at time t + tau )

taken directly from melody-extraction's **frame-level f0 track** — no note segmentation, no
12-bin rounding — folded to one octave at `n_bins` resolution (typically 30-40 cents per
bin, i.e. finer than a semitone). What that surface encodes is precisely what the swar
string throws away:

* **the diagonal** is the pitch distribution itself, at shruti resolution — a komal rishabh
  sung at 175 cents shows up at 175 cents, not as "R";
* **the off-diagonal ridges** are the transitions, and their *width and shape* carry the
  meend (a slow glide smears the ridge) and the gamak (an oscillation puts mass symmetric
  about the diagonal). A note-level bigram matrix can only say "r followed by G".

The cost is that there is no way to write a mukhyanga phrase as a surface, so unlike
M1-M7 this method **cannot use the libmogra database at all** — each raag's reference
surface is the mean of its train clips. It is therefore the honest measure of what the
symbolic/pitch space gives you *without* the mukhyanga prior, and its per-fold refit is
handled by the CV harness like any other fitted method.
"""

import numpy as np

from utils.raagdb import dataset_raags
from . import Method

EPS = 1e-12


def melody_surface(f0_hz, voiced, hop_seconds, tonic_hz, n_bins=40, tau=0.3,
                   smooth=1.0, power=0.5, max_cents_dev=2400.0):
    """(n_bins, n_bins) surface of pitch(t) vs pitch(t+tau), octave-folded, from raw frames."""
    f0 = np.asarray(f0_hz, dtype=float)
    ok = np.asarray(voiced, dtype=bool) & (f0 > 0)
    if ok.sum() < 10:
        return np.zeros((n_bins, n_bins))
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.clip(f0, 1e-9, None) / tonic_hz)
    ok &= np.abs(cents) <= max_cents_dev

    lag = max(int(round(tau / hop_seconds)), 1)
    if lag >= len(cents):
        return np.zeros((n_bins, n_bins))
    a_ok, b_ok = ok[:-lag], ok[lag:]
    both = a_ok & b_ok
    if both.sum() < 10:
        return np.zeros((n_bins, n_bins))

    scale = n_bins / 1200.0
    ia = np.floor((cents[:-lag][both] % 1200.0) * scale).astype(int) % n_bins
    ib = np.floor((cents[lag:][both] % 1200.0) * scale).astype(int) % n_bins
    H = np.zeros((n_bins, n_bins))
    np.add.at(H, (ia, ib), 1.0)

    if smooth > 0:  # circular Gaussian blur, so bin edges are not a hard boundary
        # the FFT round-trip leaves ~1e-17 negatives, and `(-1e-17) ** 0.5` is NaN, which
        # then propagates through the whole surface
        H = np.maximum(_circular_blur(H, smooth), 0.0)
    if power != 1.0:  # compress dynamic range: long nyas notes otherwise swamp everything
        H = H ** power
    n = H.sum()
    return H / n if n > 0 else H


def _circular_blur(H, sigma):
    n = H.shape[0]
    d = np.arange(n)
    d = np.minimum(d, n - d)
    k = np.exp(-0.5 * (d / sigma) ** 2)
    k /= k.sum()
    K = np.fft.rfft(k)
    out = np.fft.irfft(np.fft.rfft(H, axis=0) * K[:, None], n=n, axis=0)
    return np.fft.irfft(np.fft.rfft(out, axis=1) * K[None, :], n=n, axis=1)


def rotate_surface(H, bins):
    """Shift the surface along both axes — a change of tonic, at sub-semitone resolution."""
    return np.roll(np.roll(H, bins, axis=0), bins, axis=1)


class TDMS(Method):
    name = "m9_tdms"
    fitted = True

    def __init__(self, n_bins=40, tau=0.3, smooth=1.0, power=0.5, metric="cosine",
                 shift_mode="none", tracker="tony", tonic_mode="video",
                 separate=None, **kw):
        super().__init__(sorted(dataset_raags()), shift_mode="none", **kw)
        self.n_bins, self.tau, self.smooth, self.power = n_bins, tau, smooth, power
        self.metric, self.tracker = metric, tracker
        # The surface is built from `tracker`'s own frames, which is usually not the
        # representation's tracker, so the tonic has to be looked up rather than taken from
        # the clip. It still has to follow the representation's *policy*: hardcoding
        # "video" here made the method silently blind to the v1 annotation.
        self.tonic_mode = tonic_mode
        # same reason as tonic_mode: this class reaches for its own cache, so it needs to
        # be told which one — otherwise it silently reads unseparated audio
        self.separate = separate
        self.refs = np.zeros((len(self.raags), n_bins, n_bins))
        self._cache = {}

    # -- surfaces ------------------------------------------------------------------------

    def surface(self, feat):
        """Surface for one clip, cached — recomputing it per fold would dominate runtime."""
        f = getattr(feat, "by_tracker", {self.tracker: feat}).get(self.tracker, feat)
        key = f.clip.clip_id
        if key not in self._cache:
            from represent import _load

            cache, _, clip_tonics, video_tonics = _load(
                self.tracker, self.tonic_mode, True, 2400.0, _TONIC_KW, self.separate
            )
            entry = cache.get(key)
            if entry is None:
                self._cache[key] = np.zeros((self.n_bins, self.n_bins))
            else:
                self._cache[key] = melody_surface(
                    entry["f0"], entry["voiced"], entry["hop"],
                    (clip_tonics.get(key) if self.tonic_mode in ("clip", "chroma_clip")
                     else video_tonics.get(f.clip.video)) or f.clip.tonic_hz,
                    n_bins=self.n_bins, tau=self.tau, smooth=self.smooth, power=self.power,
                )
        return self._cache[key]

    def fit(self, feats):
        idx = {r: i for i, r in enumerate(self.raags)}
        acc = np.zeros_like(self.refs)
        n = np.zeros(len(self.raags))
        for f in feats:
            i = idx.get(f.clip.raag)
            if i is None:
                continue
            H = self.surface(f)
            if H.sum() > 0:
                acc[i] += H
                n[i] += 1
        for i in range(len(self.raags)):
            if n[i] > 0:
                acc[i] /= n[i]
        self.refs = acc
        return self

    def score_at(self, feat, k):
        H = self.surface(feat)
        if H.sum() <= 0:
            return np.zeros(len(self.raags))
        if k:
            H = rotate_surface(H, int(round(k * self.n_bins / 12)))
        x = H.reshape(-1)
        R = self.refs.reshape(len(self.raags), -1)
        if self.metric == "cosine":
            return (R @ x) / (np.linalg.norm(R, axis=1) * np.linalg.norm(x) + EPS)
        if self.metric == "dot":
            return R @ x
        if self.metric == "chi2":  # symmetric chi-square distance, negated
            return -np.sum((R - x[None, :]) ** 2 / (R + x[None, :] + EPS), axis=1)
        # negative L2, so higher is still better
        return -np.linalg.norm(R - x[None, :], axis=1)


_TONIC_KW = (("alpha", 0.6), ("beta", 0.9), ("gamma", 0.05), ("median_target", 6.0))


class TDMSPlus(Method):
    """M9 fused with a DB-based method — data-driven contour evidence plus prior knowledge.

    The two are near-orthogonal by construction: TDMS knows nothing about mukhyanga and the
    grammar methods know nothing about sub-semitone pitch. Scores are z-scored per method
    over train before adding, since a cosine and a log-likelihood are not on one scale.
    """

    name = "m9_tdms_plus"
    fitted = True

    def __init__(self, w_tdms=1.0, base="m7", base_kw=None, tdms_kw=None,
                 tdms_cls="m9", calibrate="none", **kw):
        from evaluate import make_method

        # `tdms_cls="m11"` swaps the 2-D surface for its 1-D marginal (a plain pitch
        # histogram), so the same fusion shell measures what the delay axis is worth.
        self.tdms = make_method(tdms_cls, **(tdms_kw or {}))
        self.base = make_method(base, **(base_kw or {}))
        super().__init__(self.tdms.raags, shift_mode="none", **kw)
        self.w_tdms = w_tdms
        self.order = [self.base.raags.index(r) for r in self.raags]
        self.stats = None
        # Per-raag hubness calibration. The M14 confusion matrix shows six labels absorbing
        # far more predictions than they own (Jaijaivanti 69 for 45 true clips, Des 69 for
        # 40) while recalling their own badly — a template that sits close to everything,
        # not a musical confusion. Standardising each raag's score distribution over train
        # removes the offset; this is the same fix M7 carried and the histogram line lost.
        self.calibrate = calibrate
        self.mu = None
        self.sigma = None

    def fit(self, feats):
        self.tdms.fit(feats)
        if self.base.fitted:
            self.base.fit(feats)
        usable = [f for f in feats if f.n_notes >= 2]
        if usable:
            A = np.stack([self.tdms.score_at(f, 0) for f in usable])
            B = np.stack([self.base.score_at(f, 0)[self.order] for f in usable])
            self.stats = (A.mean(), A.std() + EPS, B.mean(), B.std() + EPS)
        if self.calibrate != "none" and usable:
            am, asd, bm, bsd = self.stats
            S = np.stack([
                self.w_tdms * (self.tdms.score_at(f, 0) - am) / asd
                + (self.base.score_at(f, 0)[self.order] - bm) / bsd
                for f in usable
            ])
            self.mu = S.mean(axis=0)
            self.sigma = (S.std(axis=0) + EPS) if self.calibrate == "zscore" \
                else np.ones(S.shape[1])
        return self

    def score_at(self, feat, k):
        a = self.tdms.score_at(feat, k)
        b = self.base.score_at(feat, k)[self.order]
        if self.stats is None:
            return self.w_tdms * a + b
        am, asd, bm, bsd = self.stats
        s = self.w_tdms * (a - am) / asd + (b - bm) / bsd
        if self.mu is None:
            return s
        return (s - self.mu) / self.sigma
