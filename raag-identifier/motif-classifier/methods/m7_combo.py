"""M7 — everything that worked, plus per-raag score calibration.

Three ingredients from M4–M6, and one new one:

1. **channel** — M5's noisy-channel HMM as the scorer (ornament/meend tolerance learned
   from train), or M3's plain grammar if `use_channel=False`.
2. **fusion** — CREPE's transcription scored alongside Tony's and added in (M4).
3. **tonic marginalisation** — a learned prior over the 12 rotations (M6).
4. **hubness calibration** — new here, and the reason this is a method rather than a
   config. The M2/M3 error analysis showed a handful of raags being predicted far more
   often than they occur: AheerBhairav, KaushikDhwani, Shree and Pilu accounted for 26 of
   92 test predictions. That is a *hub* problem, not a music problem — those raags' models
   simply assign higher scores to everything, because their scales are large or their
   grammars are flat. Standardising each raag's score against the mean and spread it
   produces across train clips removes the offset without touching the ranking within a
   raag:

       score'(r) = (score(r) - mu_r) / sigma_r

   mu_r and sigma_r come only from train clips (any raag's), so nothing about the test
   labels is used; the CV harness refits them per fold like every other learned quantity.
"""

import numpy as np

from . import Method

EPS = 1e-9


class Combo(Method):
    name = "m7_combo"
    fitted = True

    def __init__(
        self,
        use_channel=True,
        w_crepe=0.0,
        calibrate="zscore",  # "zscore" | "mean" | "none"
        marginalise_tonic=False,
        temperature=1.0,
        prior_smoothing=1.0,
        base_kw=None,
        channel_kw=None,
        **kw,
    ):
        from evaluate import make_method

        if use_channel:
            self.core = make_method("m5", **(channel_kw or {}))
        else:
            # merge, don't splat: base_kw overrides these defaults, and `dict(a=1, **{"a": 2})`
            # is a TypeError rather than an override
            defaults = dict(w_arohana=1.0, lam_bi=0.7, lam_uni=0.1, uni_from_scale=0.75,
                            w_dur=1.0, w_skip=0.5, shift_mode="none")
            self.core = make_method("m3", **{**defaults, **(base_kw or {})})
        super().__init__(self.core.raags, shift_mode="none", **kw)

        self.w_crepe = w_crepe
        self.calibrate = calibrate
        self.marginalise_tonic = marginalise_tonic
        self.temperature = temperature
        self.prior_smoothing = prior_smoothing
        self.log_prior = np.log(np.full(12, 1.0 / 12.0))
        self.mu = np.zeros(len(self.raags))
        self.sigma = np.ones(len(self.raags))

    # -- scoring, before calibration ----------------------------------------------------

    def _raw_at(self, feat, k):
        by = getattr(feat, "by_tracker", None)
        if by is None:
            return self.core.score_at(feat, k)
        s = self.core.score_at(by[self.core_primary], k)
        if self.w_crepe and "crepe" in by and by["crepe"].n_notes >= 2:
            s = s + self.w_crepe * self.core.score_at(by["crepe"], k)
        return s

    core_primary = "tony"

    def _raw(self, feat):
        if not self.marginalise_tonic:
            return self._raw_at(feat, 0)
        s = np.stack([self._raw_at(feat, k) for k in range(12)]) / max(self.temperature, EPS)
        s = s + self.log_prior[:, None]
        mx = s.max(axis=0, keepdims=True)
        return mx[0] + np.log(np.exp(s - mx).sum(axis=0) + EPS)

    # -- fitting -------------------------------------------------------------------------

    def fit(self, feats):
        usable = [f for f in feats if f.n_notes >= 2]
        if self.core.fitted:
            # the channel HMM learns from the primary tracker's sequences
            self.core.fit([getattr(f, "by_tracker", {"tony": f})["tony"] for f in usable])

        if self.marginalise_tonic:
            idx = {r: i for i, r in enumerate(self.raags)}
            counts = np.full(12, self.prior_smoothing)
            for f in usable:
                if f.clip.raag not in idx:
                    continue
                col = np.array([self._raw_at(f, k)[idx[f.clip.raag]] for k in range(12)])
                col = (col / max(self.temperature, EPS))
                col -= col.max()
                p = np.exp(col)
                counts += p / (p.sum() + EPS)
            self.log_prior = np.log(counts / counts.sum())

        if self.calibrate != "none" and usable:
            S = np.stack([self._raw(f) for f in usable])  # (N, R)
            self.mu = S.mean(axis=0)
            self.sigma = S.std(axis=0) + EPS if self.calibrate == "zscore" else np.ones(len(self.raags))
        return self

    def score_at(self, feat, k):
        if feat.n_notes < 2:
            return np.zeros(len(self.raags))
        s = self._raw(feat)
        if self.calibrate == "none":
            return s
        return (s - self.mu) / self.sigma
