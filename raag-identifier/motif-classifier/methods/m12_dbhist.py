"""M12 — the pitch histogram, with the database as a prior on it.

M11 learns each raag's pitch distribution from training clips and knows nothing else. M10
tried the opposite — a template built purely from `vaadi`/`samvaadi` — and failed. The
reason M10 failed is specific and worth stating, because it points straight at the fix:

    Bageshree   vaadi=m  samvaadi=S      scale = S R g m P D n
    Bheempalasi vaadi=m  samvaadi=S      scale = S R g m P D n

Identical scale, identical vaadi, identical samvaadi. No method built on those fields can
tell them apart. But their **mukhyanga phrase statistics** are not identical at all —

    Bageshree    S .24  m .18  D .16  n .16  g .13     (P barely appears)
    Bheempalasi  S .23  n .23  P .17  g .11  m .11     (P is central)

L1 distance 0.43, and it is the textbook distinction: Bageshree weakens P and leans on D,
Bheempalasi leans on P. So the database *does* know these two apart — through how often each
swar appears across its phrases, which is exactly a pitch histogram, and which is the one
view of the DB that M1-M7 never used directly.

The template is therefore built from phrase occupancy, spread onto the same continuous bin
grid M11 uses, and blended:

    ref = (1 - lam) * learned  +  lam * db

`lam=0` is M11; `lam=1` is a purely prescriptive histogram that needs no training data at
all. Anything in between is the DB acting as a prior — which is most useful exactly where
the training data is thinnest.
"""

import numpy as np

from raagdb import dataset_raags
from . import Method
from .m11_histogram import HistogramFingerprint, fold_histogram

EPS = 1e-12


def db_histogram(raag, n_bins=120, sigma_cents=35.0, w_phrase=1.0, w_scale=0.15,
                 w_nyas=0.5, w_vaadi=0.5, w_samvaadi=0.25):
    """A raag's expected pitch distribution, read off the database.

    Mass comes mostly from **how often each swar occurs across the mukhyanga phrases plus
    aaroha/avaroha** — the part that separates same-scale raags. The vaadi/samvaadi/nyas
    boosts are small corrections on top, not the main signal (M10 is the experiment showing
    they cannot carry it alone). `w_scale` puts a floor under every legal swar so a raag is
    not certain a note never happens just because the DB did not spell it out.

    Spread onto the continuous grid with a Gaussian of `sigma_cents`, because the learned
    histograms it is blended with are smooth and a spike would not compare against them.
    """
    uni = np.zeros(12)
    for p in raag.phrases:
        for s in p:
            uni[s] += w_phrase
    for seq in (raag.aaroha, raag.avaroha):
        for s in seq:
            uni[s] += w_phrase
    for s in raag.scale:
        uni[s] += w_scale * max(uni.sum(), 1.0) / 12.0
    for s in raag.nyas:
        uni[s] += w_nyas * max(uni.sum(), 1.0) / 12.0
    for field, w in ((raag.vaadi, w_vaadi), (raag.samvaadi, w_samvaadi)):
        if field is not None:
            uni[field[0]] += w * max(uni.sum(), 1.0) / 12.0
    uni = uni / max(uni.sum(), EPS)

    centres = np.arange(12) * 100.0
    grid = (np.arange(n_bins) + 0.5) * (1200.0 / n_bins)
    d = np.abs(grid[:, None] - centres[None, :])
    d = np.minimum(d, 1200.0 - d)  # circular
    K = np.exp(-0.5 * (d / max(sigma_cents, 1e-6)) ** 2)
    H = K @ uni
    return H / max(H.sum(), EPS)


class DBHistogram(HistogramFingerprint):
    name = "m12_dbhist"
    fitted = True

    def __init__(self, lam=0.5, sigma_cents=35.0, db_kw=None, **kw):
        super().__init__(**kw)
        self.lam = lam
        db = dataset_raags()
        self.db_refs = np.stack([
            db_histogram(db[r], n_bins=self.n_bins, sigma_cents=sigma_cents,
                         **(db_kw or {}))
            for r in self.raags
        ])
        if lam >= 1.0:  # no training data needed at all
            self.refs = self.db_refs.copy()
            self.fitted = False

    def fit(self, feats):
        if self.lam >= 1.0:
            return self
        super().fit(feats)
        learned = self.refs / np.maximum(self.refs.sum(axis=1, keepdims=True), EPS)
        self.refs = (1.0 - self.lam) * learned + self.lam * self.db_refs
        return self
