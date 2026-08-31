"""Grading a method by *how wrong* it is, not just whether it is wrong.

Top-1 over 50 classes is a brutally coarse instrument for this problem, and it hides the
thing we actually want to know: when the model misses, does it miss into the right
neighbourhood? Confusing Tilak Kamod with Des is a near-miss a listener might make; confusing
it with Bairagi means the model learned nothing. Both cost exactly 1.0 under accuracy.

Everything here grades against `raagspace.affinity()`, which is built purely from the
libmogra database — so these are *prior-knowledge* metrics, not fitted ones.

## The metrics

**`mistake_affinity`** — mean affinity(true, predicted) over the **errors only**. Reported
next to `mistake_affinity_chance`, the value a uniformly random wrong guess would score.
Above chance means the mistakes are principled.

**`expected_affinity` (MEA)** — the same idea against the whole soft output rather than the
argmax: `Σ_r p(r) · affinity(true, r)`. This is the metric that rewards the user's example —
truth Bhoop with mass on Durga / Deshkar / HansDhwani scores well; the same mass on Sohani /
Bhairav / Madhukauns does not.

**`affinity_ce`** — a proper loss. Build a soft target `q(r) ∝ affinity(true, r) ** gamma`
(peaked on the true raag, with the rest of its mass on genuinely related raags) and take the
cross-entropy `-Σ_r q(r) log p(r)`. Lower is better. Unlike plain NLL this does not punish a
model for putting mass on Des when the answer is Tilak Kamod.

**`tonic_explained`** / **`rot_affinity`** — of the errors, the fraction whose predicted raag
is a near-exact *rotation* of the true raag's scale (rotational affinity ≥ `rot_threshold`,
at k ≠ 0, while direct affinity is low), and the threshold-free version, mean rotational
affinity over errors. These are not raag mistakes at all: the model read the melody and put
Sa in the wrong place. Bhoopali → Malkauns looks catastrophic by direct affinity (0.05) and
is in fact the *identical pitch set* rotated by 4 semitones. Both are reported against their
own chance rate, because rotating a 5- or 7-note scale hits something plausible far more
often than intuition suggests.

Note this measures something narrower than the oracle-tonic experiment in `ceilings.py`.
That one asks "would the true raag have won with the right Sa"; this one asks "is the raag
we picked instead literally the true one transposed". The first is the bigger number.

## Turning scores into probabilities

The methods emit wildly different score scales — M1 emits a fraction in [0,1], M3 a
log-likelihood per note, M7 a z-score. Comparing their soft outputs requires putting them on
a common footing, so each method gets a **softmax temperature calibrated on train** by
minimising ordinary NLL of the true label (standard temperature scaling). That is fit on
train only and applied unchanged to test.
"""

import numpy as np

from .raagspace import affinity

EPS = 1e-12


def softmax(scores, T):
    z = np.asarray(scores, dtype=float) / max(T, EPS)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + EPS)


def calibrate_temperature(rows, labels, grid=None):
    """Temperature scaling: pick T minimising NLL of the true label on these rows."""
    idx = {r: i for i, r in enumerate(labels)}
    S = np.array([r["scores"] for r in rows if "scores" in r])
    y = np.array([idx[r["true"]] for r in rows if "scores" in r and r["true"] in idx])
    if len(S) == 0:
        return 1.0
    S = S[: len(y)]
    grid = grid if grid is not None else np.exp(np.linspace(np.log(1e-3), np.log(1e3), 121))
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        p = softmax(S, T)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + EPS))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T


def musical_metrics(rows, temperature=1.0, gamma=4.0, rot_threshold=0.85,
                    dataset_labels=None):
    """All of the above for one method's per-clip rows (which must carry `scores`)."""
    labels, A, A_rot, best_k = affinity(names=dataset_labels)
    idx = {r: i for i, r in enumerate(labels)}
    rows = [r for r in rows if r["true"] in idx and r["pred"] in idx]
    if not rows:
        return {}
    R = len(labels)

    # --- hard: affinity of the top-1 mistake -----------------------------------------
    errs = [r for r in rows if r["true"] != r["pred"]]
    mis = float(np.mean([A[idx[r["true"]], idx[r["pred"]]] for r in errs])) if errs else float("nan")
    # chance = a uniformly random *wrong* label, averaged over the true labels we actually saw
    chance_mis = float(
        np.mean([(A[idx[r["true"]]].sum() - 1.0) / (R - 1) for r in errs])
    ) if errs else float("nan")

    # --- soft: the whole ranking -------------------------------------------------------
    have = [r for r in rows if "scores" in r]
    mea = ce = nll = float("nan")
    mea_chance = float(np.mean([A[idx[r["true"]]].mean() for r in have])) if have else float("nan")
    if have:
        S = np.array([r["scores"] for r in have])
        y = np.array([idx[r["true"]] for r in have])
        p = softmax(S, temperature)
        mea = float(np.mean((p * A[y]).sum(axis=1)))
        q = A[y] ** gamma
        q = q / q.sum(axis=1, keepdims=True)
        ce = float(np.mean(-(q * np.log(p + EPS)).sum(axis=1)))
        nll = float(np.mean(-np.log(p[np.arange(len(y)), y] + EPS)))

    # --- how many errors are really tonic errors ---------------------------------------
    def is_rot(i, j):
        return A_rot[i, j] >= rot_threshold and best_k[i, j] != 0 and A[i, j] < 0.6

    tonic = float(np.mean([is_rot(idx[r["true"]], idx[r["pred"]]) for r in errs])) if errs else float("nan")
    # threshold-free companion: how rotatable the errors are on average
    rot_aff = float(np.mean([A_rot[idx[r["true"]], idx[r["pred"]]] for r in errs])) if errs else float("nan")
    rot_aff_chance = float(
        np.mean([(A_rot[idx[r["true"]]].sum() - A_rot[idx[r["true"]], idx[r["true"]]]) / (R - 1)
                 for r in errs])
    ) if errs else float("nan")
    # chance rate: how often a random wrong label would look like a rotation
    rate = []
    for r in errs:
        i = idx[r["true"]]
        rate.append(np.mean([is_rot(i, j) for j in range(R) if j != i]))
    tonic_chance = float(np.mean(rate)) if rate else float("nan")

    return {
        "mistake_affinity": mis,
        "mistake_affinity_chance": chance_mis,
        "expected_affinity": mea,
        "expected_affinity_chance": mea_chance,
        "affinity_ce": ce,
        "nll": nll,
        "tonic_explained": tonic,
        "tonic_explained_chance": tonic_chance,
        "rot_affinity": rot_aff,
        "rot_affinity_chance": rot_aff_chance,
        "temperature": temperature,
        "n_errors": len(errs),
    }


def worst_and_best_mistakes(rows, n=6, dataset_labels=None):
    """The most and least defensible top-1 errors, for the write-up."""
    labels, A, A_rot, best_k = affinity(names=dataset_labels)
    idx = {r: i for i, r in enumerate(labels)}
    errs = [
        (A[idx[r["true"]], idx[r["pred"]]], r["true"], r["pred"],
         A_rot[idx[r["true"]], idx[r["pred"]]], best_k[idx[r["true"]], idx[r["pred"]]])
        for r in rows
        if r["true"] != r["pred"] and r["true"] in idx and r["pred"] in idx
    ]
    errs.sort()
    return errs[-n:][::-1], errs[:n]
