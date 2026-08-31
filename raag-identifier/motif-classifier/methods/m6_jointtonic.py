"""M6 — joint tonic and raag, with the rotation prior learned from train.

M3's `shift_mode="global"` already tried "let the raag model pick the tonic", by taking the
max over the 12 rotations. It lost badly (0.089 -> 0.074): a hard max hands the model twelve
independent chances to find a spurious fit, and the winner is usually spurious, because the
estimated tonic is right far more often than any single wrong rotation.

The fix is not to choose between "trust the tonic" and "search the tonic" — it is to put a
**prior** on the rotation and marginalise:

    P(raag | clip)  ∝  Σ_k  P(k) · P(clip | raag, rotated by k) ** (1/temperature)

`P(k)` is estimated on train as the posterior-weighted frequency of each rotation under the
*true* raag, so it measures how often melody-extraction's tonic is off, and by how much —
including whether the +7 (tanpura Pa) hypothesis is real or a diagnostic artefact. The two
earlier behaviours are the endpoints of this one: a prior concentrated on k=0 reproduces
`shift_mode="none"`, a flat prior with temperature -> 0 reproduces `shift_mode="global"`.
Tuning is free to land anywhere between, and to say the answer is "trust the tonic".

`temperature` matters because the base scores are length-normalised log-likelihoods: without
it the log-sum-exp is dominated by whichever rotation happens to peak, i.e. the hard max again.
"""

import numpy as np

from . import Method

EPS = 1e-12


class JointTonic(Method):
    name = "m6_jointtonic"
    fitted = True

    def __init__(self, base="m3", base_kw=None, temperature=1.0, prior_smoothing=1.0,
                 learn_prior=True, **kw):
        from evaluate import make_method

        base_kw = dict(base_kw or {})
        base_kw["shift_mode"] = "none"
        self.base = make_method(base, **base_kw)
        super().__init__(self.base.raags, shift_mode="none", **kw)
        self.temperature = temperature
        self.prior_smoothing = prior_smoothing
        self.learn_prior = learn_prior
        self.log_prior = np.log(np.full(12, 1.0 / 12.0))
        self.fitted = True

    def _rotation_scores(self, feat):
        """(12, R) matrix of base scores, one row per candidate rotation."""
        return np.stack([self.base.score_at(feat, k) for k in range(12)])

    def fit(self, feats):
        if self.base.fitted:
            self.base.fit(feats)
        if not self.learn_prior:
            return self
        idx = {r: i for i, r in enumerate(self.raags)}
        counts = np.full(12, self.prior_smoothing)
        for f in feats:
            if f.n_notes < 2 or f.clip.raag not in idx:
                continue
            col = self._rotation_scores(f)[:, idx[f.clip.raag]] / max(self.temperature, EPS)
            col = col - col.max()
            p = np.exp(col)
            counts += p / (p.sum() + EPS)  # soft vote, so a near-tie doesn't count as certainty
        self.log_prior = np.log(counts / counts.sum())
        return self

    def score_at(self, feat, k):
        if feat.n_notes < 2:
            return np.zeros(len(self.raags))
        s = self._rotation_scores(feat) / max(self.temperature, EPS)  # (12, R)
        s = s + self.log_prior[:, None]
        mx = s.max(axis=0, keepdims=True)
        return mx[0] + np.log(np.exp(s - mx).sum(axis=0) + EPS)  # log-sum-exp over rotations
