"""M11 — a pitch histogram and nothing else. No motifs, no grammar, no database.

The oldest idea in raga recognition (Chordia & Rae 2007 classify on pitch-class
distributions alone) and the right floor to measure everything else against: a clip is
just *how long it spends on each pitch*, relative to Sa, folded to one octave. Two raags
with the same swars and different phrases are indistinguishable here **by construction** —
that is the point. Whatever M3-M9 earn above this line is what phrase structure is worth.

It is also the exact ablation of M9. M9's melody surface is a 2-D histogram of pitch(t)
against pitch(t+tau); collapse the delay axis and you get this. The gap between M11 and M9
is therefore a clean measurement of what *melodic motion* adds over *pitch occupancy*.

Two knobs matter:

  n_bins   12 rounds to semitones and throws shruti away; 60-120 keeps the continuous
           pitch distribution, so a komal rishabh sung at 175 cents stays where it was
           sung. Since v1 gives a true tonic, those bin positions are finally meaningful.
  source   "frames" weights by time spent (raw f0 frames, no segmentation); "notes" uses
           the note events, weighted by duration. Frames are the purer statement of the
           idea — no segmenter in the loop.

Templates are the per-raag mean of the training clips' histograms, so unlike M1-M4 this
learns from data; it is a *baseline*, not a prescriptive method, and the CV harness refits
it per fold.
"""

import numpy as np

from raagdb import dataset_raags
from . import Method

EPS = 1e-12


def fold_histogram(cents, weights=None, n_bins=120, smooth=1.0, power=0.5):
    """Octave-folded pitch histogram, circularly smoothed and dynamic-range compressed.

    `smooth` blurs across bins so that two performances of the same swar, tuned a few
    cents apart, still overlap. `power < 1` stops a single long nyas note from swamping
    every other swar — the same compression M9 applies to its surface, kept identical so
    the M11/M9 comparison isolates the delay axis and nothing else.
    """
    c = np.asarray(cents, dtype=float)
    ok = np.isfinite(c)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        ok &= np.isfinite(w)
        w = w[ok]
    else:
        w = None
    c = c[ok]
    if c.size < 5:
        return np.zeros(n_bins)
    idx = np.floor((c % 1200.0) * (n_bins / 1200.0)).astype(int) % n_bins
    H = np.zeros(n_bins)
    np.add.at(H, idx, 1.0 if w is None else w)
    if smooth > 0:
        d = np.arange(n_bins)
        d = np.minimum(d, n_bins - d)
        kern = np.exp(-0.5 * (d / smooth) ** 2)
        H = np.real(np.fft.ifft(np.fft.fft(H) * np.fft.fft(kern / kern.sum())))
        H = np.maximum(H, 0.0)
    if power != 1.0:
        H = H ** power
    tot = H.sum()
    return H / tot if tot > 0 else H


class HistogramFingerprint(Method):
    name = "m11_histogram"
    fitted = True

    def __init__(self, n_bins=120, source="frames", tracker=None, metric="cosine",
                 smooth=1.0, power=0.5, tonic_mode="true", separate=None,
                 shift_mode="none", **kw):
        super().__init__(sorted(dataset_raags()), shift_mode=shift_mode, **kw)
        self.n_bins, self.source, self.metric = n_bins, source, metric
        self.smooth, self.power = smooth, power
        self.tracker, self.tonic_mode, self.separate = tracker, tonic_mode, separate
        self.refs = np.zeros((len(self.raags), n_bins))
        self._cache = {}

    # -- per-clip fingerprint ------------------------------------------------------------

    def histogram(self, feat):
        f = getattr(feat, "by_tracker", {self.tracker: feat}).get(self.tracker or "", feat)
        key = f.clip.clip_id
        if key in self._cache:
            return self._cache[key]

        if self.source == "notes":
            H = fold_histogram(f.clip.cents or [], None, self.n_bins, self.smooth, self.power)
        else:
            # frames: read the raw f0 track, exactly as M9 does, so the only difference
            # between this and M9 is the missing delay axis
            from represent import _load

            tracker = self.tracker or f.clip.__dict__.get("_tracker") or "crepe"
            cache, _, clip_t, video_t = _load(
                tracker, self.tonic_mode, True, 2400.0,
                (("alpha", 0.6), ("beta", 0.9), ("gamma", 0.05), ("median_target", 6.0)),
                self.separate,
            )
            entry = cache.get(key)
            if entry is None:
                H = np.zeros(self.n_bins)
            else:
                tonic = (clip_t.get(key) if self.tonic_mode in ("clip", "chroma_clip")
                         else video_t.get(f.clip.video)) or f.clip.tonic_hz
                f0, voiced = entry["f0"], entry["voiced"]
                ok = voiced & (f0 > 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cents = 1200.0 * np.log2(np.clip(f0[ok], 1e-9, None) / tonic)
                cents = cents[np.abs(cents) <= 2400.0]
                H = fold_histogram(cents, None, self.n_bins, self.smooth, self.power)
        self._cache[key] = H
        return H

    # -- templates -----------------------------------------------------------------------

    def fit(self, feats):
        idx = {r: i for i, r in enumerate(self.raags)}
        acc = np.zeros((len(self.raags), self.n_bins))
        n = np.zeros(len(self.raags))
        for f in feats:
            i = idx.get(f.clip.raag)
            if i is None:
                continue
            H = self.histogram(f)
            if H.sum() > 0:
                acc[i] += H
                n[i] += 1
        self.refs = acc / np.maximum(n, 1)[:, None]
        return self

    def score_at(self, feat, k):
        H = self.histogram(feat)
        if H.sum() <= 0:
            return np.zeros(len(self.raags))
        if k:
            H = np.roll(H, int(round(k * self.n_bins / 12)))
        R = self.refs
        if self.metric == "cosine":
            return (R @ H) / (np.linalg.norm(R, axis=1) * np.linalg.norm(H) + EPS)
        if self.metric == "dot":
            return R @ H
        if self.metric == "chi2":  # symmetric chi-square distance, negated
            return -np.sum((R - H[None, :]) ** 2 / (R + H[None, :] + EPS), axis=1)
        return -np.linalg.norm(R - H[None, :], axis=1)
