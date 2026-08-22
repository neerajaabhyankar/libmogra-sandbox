"""M4 — CREPE, and Tony + CREPE fused.

The two trackers fail differently, which is the only reason to carry both.

* **Tony** is pYIN plus a note-level HMM with onset sensitivity and duration pruning. It
  emits few, confident, well-separated notes (median 13 per clip) and throws ornaments away.
  It is the best single source in this pipeline by a factor of two.
* **CREPE** is a frame-level deep pitch tracker with no note model at all; here its frames
  are segmented by melody-extraction's `segment_notes`. It keeps the fast material Tony's
  HMM prunes — kan-swars, the interior of a meend — at the cost of a lot of noise.

So the hypothesis worth testing is not "is CREPE better" (it isn't) but "does CREPE see
phrase evidence Tony deleted". Fusion is a weighted sum of the two length-normalised
grammar log-likelihoods, which is the correct combination if the two transcriptions were
conditionally independent given the raag. They aren't — same audio, same tonic — so `w_crepe`
is tuned rather than assumed.
"""

import numpy as np

from . import Method
from .m3_grammar import GrammarMatcher


class TrackerFusion(Method):
    name = "m4_fusion"

    def __init__(self, shift_mode="none", w_crepe=0.5, primary="tony", secondary="crepe",
                 grammar_kw=None, **kw):
        grammar_kw = dict(
            w_arohana=1.0, lam_bi=0.7, lam_uni=0.1, uni_from_scale=0.75,
            w_dur=1.0, w_skip=0.5, **(grammar_kw or {})
        )
        base = GrammarMatcher(shift_mode="none", **grammar_kw)
        super().__init__(base.raags, shift_mode=shift_mode, **kw)
        self.base = base
        self.w_crepe = w_crepe
        self.primary, self.secondary = primary, secondary

    def score_at(self, feat, k):
        by = getattr(feat, "by_tracker", None)
        if by is None:  # plain single-tracker features: nothing to fuse
            return self.base.score_at(feat, k)
        s = self.base.score_at(by[self.primary], k)
        if self.w_crepe and self.secondary in by:
            other = by[self.secondary]
            if other.n_notes >= 2:
                s = s + self.w_crepe * self.base.score_at(other, k)
        return s


class SingleTracker(TrackerFusion):
    """CREPE alone, for the ablation that says whether fusion is doing anything."""

    name = "m4_single"

    def __init__(self, tracker="crepe", **kw):
        super().__init__(w_crepe=0.0, primary=tracker, **kw)
