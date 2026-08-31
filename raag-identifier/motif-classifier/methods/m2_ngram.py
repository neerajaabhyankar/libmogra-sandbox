"""M2 — IDF-weighted n-gram / skip-gram phrase overlap, plus a scale term.

Three fixes to M1, each aimed at one of its failure modes:

* **Partial credit.** A phrase is broken into its 2/3/4-grams, so hitting three notes of
  `nDPmGm` still counts. M1 scored that zero.
* **Specificity.** Every n-gram is weighted by IDF over the 117-raag database. `mP` occurs
  in 66 raags and carries almost no information; `,d,nSgm` occurs in one.
* **Ornament tolerance.** The clip side contributes skip-grams (see `features`), so a
  kan-swar or a tracker glitch inside a phrase does not destroy the match.

On top of the phrase term sits a scale term — how much of the clip's sung duration lands
on swars the raag actually contains. It is crude but it is dense, and on a 6 s clip with no
recognisable motif it is the only evidence there is.
"""

import numpy as np

from utils.raagdb import collapse, dataset_raags, ngram_document_frequency
from . import Method


class NgramPhraseMatcher(Method):
    name = "m2_ngram"

    def __init__(
        self,
        shift_mode="global",
        n_min=2,
        n_max=4,
        idf_power=1.0,
        w_mukhyanga=1.0,
        w_arohana=0.3,
        len_power=1.0,
        tf_saturate=True,
        norm="raag_l1",
        w_phrase=1.0,
        w_scale=1.0,
        vivadi_penalty=1.0,
        w_nyas=0.0,
        **kw,
    ):
        raags = dataset_raags()
        super().__init__(raags.keys(), shift_mode=shift_mode, **kw)
        self.n_min, self.n_max = n_min, n_max
        self.tf_saturate = tf_saturate
        self.w_phrase, self.w_scale = w_phrase, w_scale
        self.vivadi_penalty, self.w_nyas = vivadi_penalty, w_nyas

        df, n_docs = ngram_document_frequency(n_min=n_min, n_max=n_max, corpus="all")
        idf = {g: (np.log(n_docs / c)) ** idf_power for g, c in df.items()}
        default_idf = float(np.log(n_docs)) ** idf_power  # an n-gram in no DB raag at all

        self.profiles = []  # per raag: {ngram: weight}
        self.norms = []
        self.scale_masks = np.zeros((len(self.raags), 12))
        self.nyas_masks = np.zeros((len(self.raags), 12))
        for i, folder in enumerate(self.raags):
            r = raags[folder]
            prof = {}
            sources = [(p, w_mukhyanga) for p in r.phrases]
            sources += [(r.aaroha, w_arohana), (r.avaroha, w_arohana)]
            for seq, w in sources:
                seq = collapse(seq)
                for n in range(n_min, n_max + 1):
                    for j in range(len(seq) - n + 1):
                        g = tuple(seq[j : j + n])
                        prof[g] = max(prof.get(g, 0.0), w * (n**len_power) * idf.get(g, default_idf))
            self.profiles.append(prof)
            vals = np.array(list(prof.values())) if prof else np.zeros(1)
            self.norms.append(
                {"raag_l1": vals.sum(), "raag_l2": np.sqrt((vals**2).sum()), "none": 1.0}[norm] or 1.0
            )
            for s in r.scale:
                self.scale_masks[i, s] = 1.0
            for s in r.nyas:
                self.nyas_masks[i, s] = 1.0
            for sv in (r.vaadi, r.samvaadi):
                if sv is not None:
                    self.nyas_masks[i, sv[0]] = 1.0

    def score_at(self, feat, k):
        clip_ngrams = {}
        for n in range(self.n_min, self.n_max + 1):
            clip_ngrams.update(feat.rot_ngrams(n, k))
        if self.tf_saturate:
            clip_ngrams = {g: np.log1p(v) for g, v in clip_ngrams.items()}

        dur = feat.rot_unigram_dur(k)
        total = dur.sum()
        p = dur / total if total > 0 else dur

        phrase = np.zeros(len(self.raags))
        for i, prof in enumerate(self.profiles):
            if not prof:
                continue
            hit = 0.0
            for g, w in prof.items():
                tf = clip_ngrams.get(g)
                if tf:
                    hit += w * tf
            phrase[i] = hit / self.norms[i]

        in_scale = self.scale_masks @ p
        scale = in_scale - self.vivadi_penalty * (1.0 - in_scale)
        nyas = self.nyas_masks @ p
        return self.w_phrase * phrase + self.w_scale * scale + self.w_nyas * nyas
