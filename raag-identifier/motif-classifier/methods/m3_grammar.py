"""M3 — the phrase inventory as a generative grammar.

M1 and M2 both ask "is this phrase present?", which on a 6-second clip is often answered by
silence: no verbatim motif, few n-gram hits, and the decision falls through to the scale
term. M3 turns the question round. Each raag's phrases + aaroha + avaroha define a
**smoothed bigram (optionally trigram) transition model over the 12 swars**, and a clip is
scored by the log-likelihood of its swar sequence under that model.

Now *every* note transition is evidence. `g R S` supports Bageshree a little; a `G` after
an `m` rules it out a lot. Nothing has to match exactly, and the arithmetic is a couple of
tensor contractions per clip, so the tonic search is free.

Smoothing is the whole design:
    P(s' | s, raag) = λ2 · bigram(raag) + λ1 · unigram(raag) + λ0 · uniform
with the raag unigram itself mixed from its phrase statistics and a flat prior over its
scale — so a transition the DB never spells out is merely unlikely, not impossible, but a
transition into a swar outside the raag is heavily punished.
"""

import numpy as np

from raagdb import collapse, dataset_raags
from . import Method

EPS = 1e-12


def _normalize_rows(m):
    s = m.sum(axis=-1, keepdims=True)
    return np.divide(m, s, out=np.zeros_like(m), where=s > 0)


class GrammarMatcher(Method):
    name = "m3_grammar"

    def __init__(
        self,
        shift_mode="global",
        w_mukhyanga=1.0,
        w_arohana=0.5,
        symmetric=False,
        lam_bi=0.7,
        lam_uni=0.25,
        lam_flat=0.05,
        uni_from_scale=0.5,
        uni_flat=0.02,
        nyas_boost=0.0,
        w_dur=0.0,
        dur_weighted=False,
        w_skip=0.0,
        w_trigram=0.0,
        length_norm=True,
        **kw,
    ):
        raags = dataset_raags()
        super().__init__(raags.keys(), shift_mode=shift_mode, **kw)
        self.w_dur, self.w_trigram, self.length_norm = w_dur, w_trigram, length_norm
        self.dur_weighted, self.w_skip = dur_weighted, w_skip

        R = len(self.raags)
        self.log_bi = np.zeros((R, 12, 12))
        self.log_uni = np.zeros((R, 12))
        self.log_tri = np.zeros((R, 12, 12, 12)) if w_trigram > 0 else None

        for i, folder in enumerate(self.raags):
            r = raags[folder]
            sources = [(collapse(p), w_mukhyanga) for p in r.phrases]
            sources += [(collapse(r.aaroha), w_arohana), (collapse(r.avaroha), w_arohana)]

            uni = np.zeros(12)
            bi = np.zeros((12, 12))
            tri = np.zeros((12, 12, 12))
            for seq, w in sources:
                for s in seq:
                    uni[s] += w
                for a, b in zip(seq, seq[1:]):
                    bi[a, b] += w
                    if symmetric:  # a phrase read backwards is still idiomatic movement
                        bi[b, a] += w
                for a, b, c in zip(seq, seq[1:], seq[2:]):
                    tri[a, b, c] += w

            scale = np.zeros(12)
            for s in r.scale:
                scale[s] = 1.0
            if nyas_boost:
                for s in r.nyas:
                    scale[s] += nyas_boost
            scale = scale / max(scale.sum(), EPS)

            uni_p = uni / max(uni.sum(), EPS)
            uni_mix = (1 - uni_from_scale) * uni_p + uni_from_scale * scale
            uni_mix = (1 - uni_flat) * uni_mix + uni_flat / 12.0
            uni_mix /= uni_mix.sum()
            self.log_uni[i] = np.log(uni_mix + EPS)

            bi_p = _normalize_rows(bi)
            bi_mix = lam_bi * bi_p + lam_uni * uni_mix[None, :] + lam_flat / 12.0
            # rows the DB says nothing about back off entirely to the unigram
            empty = bi.sum(axis=1) == 0
            bi_mix[empty] = (lam_bi + lam_uni) * uni_mix[None, :] + lam_flat / 12.0
            bi_mix = _normalize_rows(bi_mix)
            self.log_bi[i] = np.log(bi_mix + EPS)

            if self.log_tri is not None:
                tri_p = _normalize_rows(tri)
                tri_mix = lam_bi * tri_p + (1 - lam_bi) * bi_mix[:, None, :]
                empty3 = tri.sum(axis=-1) == 0
                tri_mix[empty3] = np.broadcast_to(bi_mix[:, None, :], tri_mix.shape)[empty3]
                self.log_tri[i] = np.log(_normalize_rows(tri_mix) + EPS)

    def score_at(self, feat, k):
        bg = feat.rot_bigram(k, "bigram_dur" if self.dur_weighted else "bigram")
        if self.w_skip:
            bg = bg + self.w_skip * feat.rot_bigram(k, "bigram_skip")
        n_trans = bg.sum()
        ll = np.tensordot(self.log_bi, bg, axes=([1, 2], [0, 1]))

        if self.w_trigram and feat.trigram:
            for (a, b, c), v in feat.trigram.items():
                a, b, c = (a + k) % 12, (b + k) % 12, (c + k) % 12
                ll += self.w_trigram * v * self.log_tri[:, a, b, c]

        if self.w_dur:
            dur = feat.rot_unigram_dur(k)
            total = dur.sum()
            if total > 0:
                ll += self.w_dur * (self.log_uni @ (dur / total)) * max(n_trans, 1.0)

        if self.length_norm and n_trans > 0:
            ll = ll / n_trans
        return ll
