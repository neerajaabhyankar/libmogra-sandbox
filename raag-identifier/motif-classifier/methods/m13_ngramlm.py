"""M13 — n-gram transition models learned from data, with the database as a prior.

M3 builds each raag's bigram model *prescriptively*, from the mukhyanga phrases in the
database. M11/M12 showed that pitch occupancy alone gets most of the way, which raises the
obvious question this method answers: **is there anything in melodic transitions that the
histogram does not already have, and does the database help or hurt there?**

Three settings from one class:

    lam_db = 0    transitions learned from the training clips only
    lam_db = 1    transitions from the database only (M3's model, re-scored here)
    0 < lam < 1   the database as a prior on the learned model

The learned model is the mean of the training clips' bigram matrices per raag, so it has
seen how these swars are *actually* strung together in performance — including transitions
the DB never spells out — while the DB half supplies structure for raags with few clips.

**Denoising matters more here than anywhere else.** A bigram over a jittery transcription
counts tracker stutter, not melody: one spurious note between S and R destroys the S->R
evidence and invents two transitions that never happened. The knobs that control it are all
in the representation, and are swept rather than assumed:

    collapse_repeats   drop consecutive duplicate swars (S S S R -> S R)
    min_dur            discard notes shorter than this many seconds
    note_source        Tony's note HMM, or re-segmenting CREPE frames
    soft_sigma         accumulate over a soft swar membership instead of hard rounding

`../source-separation` is the other half of denoising: with the tabla removed the note
segmenter has far less to trip over (jitter drops ~10x on HPSS-separated audio).
"""

import numpy as np

from utils.raagdb import dataset_raags
from . import Method

EPS = 1e-12


def _rownorm(m):
    s = m.sum(axis=-1, keepdims=True)
    return np.divide(m, s, out=np.full_like(m, 1.0 / m.shape[-1]), where=s > 0)


class NgramLM(Method):
    name = "m13_ngramlm"
    fitted = True

    def __init__(self, lam_db=0.3, order=2, which="bigram", w_uni=0.3, lam_flat=0.02,
                 length_norm=True, db_kw=None, shift_mode="none", **kw):
        super().__init__(sorted(dataset_raags()), shift_mode=shift_mode, **kw)
        self.lam_db, self.order, self.which = lam_db, order, which
        self.w_uni, self.lam_flat, self.length_norm = w_uni, lam_flat, length_norm
        R = len(self.raags)
        self.bi = np.full((R, 12, 12), 1.0 / 12.0)
        self.uni = np.full((R, 12), 1.0 / 12.0)
        self.tri = None
        self.db_bi, self.db_uni = self._db_models(db_kw or {})
        if lam_db >= 1.0:
            self.bi, self.uni = self.db_bi.copy(), self.db_uni.copy()
            self.fitted = False

    def _db_models(self, db_kw):
        """Transition + unigram counts read straight off the phrase inventory."""
        from utils.raagdb import collapse

        raags = dataset_raags()
        R = len(self.raags)
        bi = np.zeros((R, 12, 12))
        uni = np.zeros((R, 12))
        w_ar = db_kw.get("w_arohana", 1.0)
        for i, name in enumerate(self.raags):
            r = raags[name]
            sources = [(collapse(p), 1.0) for p in r.phrases]
            sources += [(collapse(r.aaroha), w_ar), (collapse(r.avaroha), w_ar)]
            for seq, w in sources:
                for a in seq:
                    uni[i, a] += w
                for a, b in zip(seq, seq[1:]):
                    bi[i, a, b] += w
            for s in r.scale:  # a legal swar is unlikely, never impossible
                uni[i, s] += db_kw.get("uni_from_scale", 0.75) * max(uni[i].sum(), 1.0) / 12.0
        return _rownorm(bi), _rownorm(uni)

    def fit(self, feats):
        if self.lam_db >= 1.0:
            return self
        idx = {r: i for i, r in enumerate(self.raags)}
        R = len(self.raags)
        bi = np.zeros((R, 12, 12))
        uni = np.zeros((R, 12))
        tri = np.zeros((R, 12, 12, 12)) if self.order >= 3 else None
        for f in feats:
            i = idx.get(f.clip.raag)
            if i is None:
                continue
            bi[i] += getattr(f, self.which)
            uni[i] += f.unigram_count
            if tri is not None:
                for g, v in f.trigram.items():
                    tri[i][g] += v
        self.bi = _rownorm((1.0 - self.lam_db) * _rownorm(bi) + self.lam_db * self.db_bi)
        self.uni = _rownorm((1.0 - self.lam_db) * _rownorm(uni) + self.lam_db * self.db_uni)
        if tri is not None:
            self.tri = _rownorm(tri)
        return self

    def score_at(self, feat, k):
        B = feat.rot_bigram(k, self.which)
        u = feat.rot_unigram_count(k)
        n = B.sum()
        if n <= 0:
            return np.zeros(len(self.raags))
        P = (1.0 - self.lam_flat) * self.bi + self.lam_flat / 12.0
        # cross-entropy of this clip's transitions under each raag's model
        s = np.tensordot(B, np.log(P + EPS), axes=([0, 1], [1, 2]))
        if self.w_uni:
            Pu = (1.0 - self.lam_flat) * self.uni + self.lam_flat / 12.0
            s = s + self.w_uni * (np.log(Pu + EPS) @ u)
        if self.order >= 3 and self.tri is not None:
            Pt = (1.0 - self.lam_flat) * self.tri + self.lam_flat / 12.0
            lt = np.log(Pt + EPS)
            s = s + sum(v * lt[:, a, b, c] for (a, b, c), v in feat.trigram.items())
        return s / n if self.length_norm else s
