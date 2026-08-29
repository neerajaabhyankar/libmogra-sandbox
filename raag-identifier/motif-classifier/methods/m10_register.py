"""M10 — emphasis and register, the two things an annotated tonic makes usable.

Every method up to M9 is octave-folded and treats a raag's swars as a set. That loses two
signals the database has been carrying all along:

  * **which swar the raag leans on.** `vaadi` and `samvaadi` are parsed in `raagdb` and used
    by exactly one method (M2), which puts them in the same binary mask as `nyas` — so
    Bhoopali (vaadi G, samvaadi D) and Deshkar (vaadi D, samvaadi G) get *identical*
    templates despite sharing a scale and differing in nothing else. Ranking the emphasis
    (vaadi > samvaadi > nyas > rest) is what separates them.
  * **register.** `phrase_octaves` and `aaroha_oct` are parsed and used nowhere. Deshkar
    lives in the upper tetrachord where Bhoopali sits lower; Marwa's phrases hang off mandra
    Ni and Dha (29 % of its phrase notes are marked non-madhya, the highest in the DB).

Both are duration-weighted statements about *where the weight sits relative to Sa*, so both
are worthless under a tonic that is off by a semitone and only become measurable with v1's
annotation. That is the point of the method: it is the one that could not have been written
before the tonic column existed.

Score = w_emph * <clip pitch-class mass, raag emphasis template>
      + w_reg  * <clip register mass, raag register template>
      - w_vivadi * (mass on swars outside the scale)
all on L1-normalised distributions, so clip length drops out.
"""

import numpy as np

from raagdb import dataset_raags
from . import Method


class RegisterMethod(Method):
    name = "m10"

    def __init__(self, w_emph=1.0, w_reg=0.5, w_vivadi=1.0,
                 vaadi_w=3.0, samvaadi_w=2.0, nyas_w=1.0, scale_w=0.5,
                 reg_prior=0.25, shift_mode="none", **kw):
        raag_db = dataset_raags()
        super().__init__(raag_db.keys(), shift_mode=shift_mode, **kw)
        self.w_emph, self.w_reg, self.w_vivadi = w_emph, w_reg, w_vivadi
        R = len(self.raags)

        # ---- emphasis template: a ranked weight per swar, not a mask
        self.emph = np.zeros((R, 12))
        self.in_scale = np.zeros((R, 12))
        for i, name in enumerate(self.raags):
            r = raag_db[name]
            for s in r.scale:
                self.emph[i, s] = scale_w
                self.in_scale[i, s] = 1.0
            for s in r.nyas:
                self.emph[i, s] = max(self.emph[i, s], nyas_w)
            if r.samvaadi is not None:
                self.emph[i, r.samvaadi[0]] = max(self.emph[i, r.samvaadi[0]], samvaadi_w)
            if r.vaadi is not None:
                self.emph[i, r.vaadi[0]] = max(self.emph[i, r.vaadi[0]], vaadi_w)
        self.emph /= np.maximum(self.emph.sum(axis=1, keepdims=True), 1e-9)

        # ---- register template: how the DB's own phrase notes distribute over mandra /
        # madhya / taar, smoothed toward uniform so a raag with few marked octaves does not
        # get an overconfident template
        self.reg = np.zeros((R, 3))
        for i, name in enumerate(self.raags):
            r = raag_db[name]
            octs = [o for po in r.phrase_octaves for o in po]
            octs += list(r.aaroha_oct) + list(r.avaroha_oct)
            for o in octs:
                self.reg[i, int(np.clip(o, -1, 1)) + 1] += 1.0
            if r.vaadi is not None:
                self.reg[i, int(np.clip(r.vaadi[1], -1, 1)) + 1] += 2.0
        self.reg += reg_prior * self.reg.sum(axis=1, keepdims=True).clip(min=1.0) / 3.0
        self.reg /= np.maximum(self.reg.sum(axis=1, keepdims=True), 1e-9)

    def score_at(self, feat, k):
        u = feat.rot_unigram_dur(k)
        tot = u.sum()
        if tot <= 0:
            return np.zeros(len(self.raags))
        p = u / tot
        rd = feat.rot_reg_dur(k)
        q = rd.sum(axis=1)
        q = q / max(q.sum(), 1e-9)
        return (self.w_emph * (self.emph @ p)
                + self.w_reg * (self.reg @ q)
                - self.w_vivadi * ((1.0 - self.in_scale) @ p))


EPS = 1e-12


class RegisterPlus(Method):
    """M10 fused with a phrase method, the same z-scored addition M9+ uses.

    The two halves answer different questions — the grammar asks "does this clip *move*
    like the raag?", the emphasis template asks "does it *rest* where the raag rests?" — so
    they should be closer to complementary than redundant.
    """

    name = "m10_register_plus"
    fitted = True

    def __init__(self, w_reg=1.0, base="m4", base_kw=None, reg_kw=None, **kw):
        from evaluate import make_method

        self.reg_m = RegisterMethod(**(reg_kw or {}))
        self.base = make_method(base, **(base_kw or {}))
        super().__init__(self.reg_m.raags, shift_mode="none", **kw)
        self.w_reg = w_reg
        self.order = [self.base.raags.index(r) for r in self.raags]
        self.stats = None

    def fit(self, feats):
        if self.base.fitted:
            self.base.fit(feats)
        usable = [f for f in feats if f.n_notes >= 2]
        if usable:
            A = np.stack([self.reg_m.score_at(f, 0) for f in usable])
            B = np.stack([self.base.score_at(f, 0)[self.order] for f in usable])
            self.stats = (A.mean(), A.std() + EPS, B.mean(), B.std() + EPS)
        return self

    def score_at(self, feat, k):
        a = self.reg_m.score_at(feat, k)
        b = self.base.score_at(feat, k)[self.order]
        if self.stats is None:
            return self.w_reg * a + b
        am, asd, bm, bsd = self.stats
        return self.w_reg * (a - am) / asd + (b - bm) / bsd
