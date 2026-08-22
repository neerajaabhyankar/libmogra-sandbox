"""M5 — noisy-channel HMM: what the singer intended vs. what the tracker wrote down.

M1–M3 all compare DB phrases to the transcription as if the transcription were the melody.
It isn't. A meend from P down to S drags the pitch through G and R, and the note segmenter
dutifully writes `P G R S` where the phrase is `P S`. A gamak around R throws off `N` and
`S` either side of it. A kan-swar is a 60 ms grace note the HMM sometimes keeps and
sometimes swallows. So the DB phrase and the transcription are not two samples of the same
alphabet — one is what was *intended*, the other is that passed through a noisy channel.

Model it as exactly that:

    hidden  h_t   the intended swar               transitions = the raag's phrase grammar
    observed o_t  the transcribed swar            emissions   = P(written | intended)

    A_r = (1 - p_self) * grammar(raag) + p_self * I
    P(o | raag) = forward algorithm over A_r and E

The self-loop is what buys insertion tolerance: while the intended note is held at P, the
hidden state can sit on P for several observations and emit the G and R that the meend
passed through, at whatever cost E says those cost. **E is estimated from the train split
by Baum–Welch** with each clip's transitions pinned to its true raag — so it is a model of
this tracker's ornament behaviour, pooled over all raags, not a model of any one raag.
That is the "let the train set tell you what's possible" part of the design: nothing about
which substitutions are plausible is hand-written.

E is shared across raags by construction, so it cannot leak raag identity — but it is still
estimated from labels, so the CV harness refits it per fold.
"""

import numpy as np

from raagdb import collapse, dataset_raags
from . import Method

EPS = 1e-12


def _norm_rows(m):
    s = m.sum(axis=-1, keepdims=True)
    return np.divide(m, s, out=np.full_like(m, 1.0 / m.shape[-1]), where=s > 0)


def raag_grammars(raags, order, w_mukhyanga, w_arohana, uni_from_scale, lam_bi, lam_uni,
                  lam_flat, uni_flat, symmetric=False):
    """The M3 grammar, factored out so M5/M6/M7 build transitions the same way M3 does."""
    R = len(order)
    A = np.zeros((R, 12, 12))
    U = np.zeros((R, 12))
    for i, folder in enumerate(order):
        r = raags[folder]
        sources = [(collapse(p), w_mukhyanga) for p in r.phrases]
        sources += [(collapse(r.aaroha), w_arohana), (collapse(r.avaroha), w_arohana)]
        uni = np.zeros(12)
        bi = np.zeros((12, 12))
        for seq, w in sources:
            for s in seq:
                uni[s] += w
            for a, b in zip(seq, seq[1:]):
                bi[a, b] += w
                if symmetric:
                    bi[b, a] += w
        scale = np.zeros(12)
        for s in r.scale:
            scale[s] = 1.0
        scale /= max(scale.sum(), EPS)
        uni_p = uni / max(uni.sum(), EPS)
        u = (1 - uni_from_scale) * uni_p + uni_from_scale * scale
        u = (1 - uni_flat) * u + uni_flat / 12.0
        u /= u.sum()
        U[i] = u
        bi_mix = lam_bi * _norm_rows(bi) + lam_uni * u[None, :] + lam_flat / 12.0
        empty = bi.sum(axis=1) == 0
        bi_mix[empty] = (lam_bi + lam_uni) * u[None, :] + lam_flat / 12.0
        A[i] = _norm_rows(bi_mix)
    return A, U


def forward_loglik(obs, A, E, pi, soft=None):
    """log P(obs | A, E, pi) for a stack of R models at once.

    A: (R,12,12)  E: (12,12) shared  pi: (R,12).  obs: list[int].
    Scaled forward recursion, so the return is already a sum of per-step log scalers.

    `soft` (T,12), if given, replaces the hard observation with a *soft* one: the per-state
    likelihood becomes `E @ w_t` — the mixture over what the note might have been — instead
    of the single column `E[:, o_t]`. This is the un-quantized path; `soft` one-hot recovers
    the hard behaviour exactly.
    """
    B = (E @ np.asarray(soft).T).T if soft is not None else E[:, obs].T  # (T, 12)
    alpha = pi * B[0][None, :]
    scale = alpha.sum(axis=1, keepdims=True)
    ll = np.log(scale[:, 0] + EPS)
    alpha = alpha / (scale + EPS)
    for t in range(1, len(B)):
        alpha = np.einsum("rh,rhg->rg", alpha, A) * B[t][None, :]
        scale = alpha.sum(axis=1, keepdims=True)
        ll = ll + np.log(scale[:, 0] + EPS)
        alpha = alpha / (scale + EPS)
    return ll


def _baum_welch_emissions(obs_by_raag, A, pi, E0, n_iter=6, prior=0.5):
    """Re-estimate the shared emission matrix E, transitions pinned per clip to its raag.

    Standard Baum-Welch, except only E is free: each clip contributes gamma_t(h) * [o_t = o]
    to a pooled count matrix. `prior` is a Dirichlet smoothing count so an unseen
    intended->written pair never goes to probability zero.
    """
    E = E0.copy()
    for _ in range(n_iter):
        counts = np.full((12, 12), prior)
        for ri, obs in obs_by_raag:
            T = len(obs)
            a, b = A[ri], pi[ri]
            alpha = np.zeros((T, 12))
            c = np.zeros(T)
            alpha[0] = b * E[:, obs[0]]
            c[0] = alpha[0].sum() + EPS
            alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ a) * E[:, obs[t]]
                c[t] = alpha[t].sum() + EPS
                alpha[t] /= c[t]
            beta = np.zeros((T, 12))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (a @ (E[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + EPS
            for t, o in enumerate(obs):
                counts[:, o] += gamma[t]
        E = _norm_rows(counts)
    return E


class ChannelGrammar(Method):
    name = "m5_channel"
    fitted = True

    def __init__(
        self,
        shift_mode="none",
        w_mukhyanga=1.0,
        w_arohana=1.0,
        lam_bi=0.7,
        lam_uni=0.1,
        lam_flat=0.05,
        uni_from_scale=0.75,
        uni_flat=0.02,
        p_self=0.35,
        e_self=0.6,
        e_neighbour=0.25,
        n_iter=6,
        prior=0.5,
        emission_temp=1.0,
        w_dur=0.0,
        length_norm=True,
        learn_emissions=True,
        **kw,
    ):
        raags = dataset_raags()
        super().__init__(raags.keys(), shift_mode=shift_mode, **kw)
        self.length_norm = length_norm
        self.n_iter, self.prior, self.learn_emissions = n_iter, prior, learn_emissions
        # A learned channel with only 0.33 on its diagonal is *very* forgiving, which costs
        # discriminative power; the temperature lets tuning sharpen it back up (<1) or
        # soften it further (>1) instead of taking Baum-Welch's answer as final.
        self.emission_temp = emission_temp
        self.w_dur = w_dur

        A, U = raag_grammars(raags, self.raags, w_mukhyanga, w_arohana, uni_from_scale,
                             lam_bi, lam_uni, lam_flat, uni_flat)
        self.A = (1 - p_self) * A + p_self * np.eye(12)[None, :, :]
        self.pi = U

        # Initial channel: mostly faithful, with the remaining mass on immediate semitone
        # neighbours — the shape a meend or a mistuned komal swar produces. Baum-Welch
        # moves it from here; with learn_emissions=False this stays the whole model.
        E = np.full((12, 12), (1.0 - e_self - e_neighbour) / 9.0)
        for h in range(12):
            E[h, h] = e_self
            E[h, (h + 1) % 12] = e_neighbour / 2
            E[h, (h - 1) % 12] = e_neighbour / 2
        self.E = _norm_rows(E)

    def fit(self, feats):
        if not self.learn_emissions:
            return self
        idx = {r: i for i, r in enumerate(self.raags)}
        obs_by_raag = [
            (idx[f.clip.raag], f.clip.swars)
            for f in feats
            if f.n_notes >= 2 and f.clip.raag in idx
        ]
        if obs_by_raag:
            E = _baum_welch_emissions(obs_by_raag, self.A, self.pi, self.E,
                                      n_iter=self.n_iter, prior=self.prior)
            if self.emission_temp != 1.0:
                E = _norm_rows(E ** (1.0 / self.emission_temp))
            self.E = E
        return self

    def score_at(self, feat, k):
        obs = feat.rot_swars(k)
        if len(obs) < 1:
            return np.zeros(len(self.raags))
        soft = feat.rot_soft(k) if getattr(feat, "soft", None) is not None else None
        ll = forward_loglik(obs, self.A, self.E, self.pi, soft=soft)
        if self.w_dur:
            dur = feat.rot_unigram_dur(k)
            total = dur.sum()
            if total > 0:
                ll = ll + self.w_dur * len(obs) * (np.log(self.pi + EPS) @ (dur / total))
        return ll / len(obs) if self.length_norm else ll
