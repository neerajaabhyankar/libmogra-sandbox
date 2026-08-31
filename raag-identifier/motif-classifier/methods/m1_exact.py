"""M1 — naive exact phrase matcher.

The literal reading of the brief: does the raag's mukhyanga phrase appear, verbatim and
contiguous, in what was sung? Score = fraction of the raag's phrases that do, with a small
bonus for the length of the longest one so that hitting `nDPmGm` beats hitting `Sm`.

Expected to be weak — it is here to show *how* it is weak, which is what motivates M2/M3.
"""

import numpy as np

from raagdb import collapse, dataset_raags
from . import Method


def _contains(haystack_str, needle_str):
    return needle_str in haystack_str


class ExactPhraseMatcher(Method):
    name = "m1_exact"

    def __init__(self, shift_mode="none", length_bonus=0.02, **kw):
        raags = dataset_raags()
        super().__init__(raags.keys(), shift_mode=shift_mode, **kw)
        self.length_bonus = length_bonus
        # phrases as strings over a 12-char alphabet, so the search is a C-level `in`
        self.phrases = []
        for folder in self.raags:
            ph = []
            for p in raags[folder].phrases:
                c = tuple(collapse(p))
                if len(c) >= 2:
                    s = "".join(chr(65 + x) for x in c)
                    if s not in ph:
                        ph.append(s)
            self.phrases.append(ph)

    def score_at(self, feat, k):
        seq = "".join(chr(65 + s) for s in feat.rot_swars(k))
        out = np.zeros(len(self.raags))
        for i, phrases in enumerate(self.phrases):
            if not phrases:
                continue
            hits = [p for p in phrases if _contains(seq, p)]
            longest = max((len(p) for p in hits), default=0)
            out[i] = len(hits) / len(phrases) + self.length_bonus * longest
        return out
