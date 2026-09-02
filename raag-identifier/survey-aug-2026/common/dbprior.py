"""The libmogra raag database, reshaped into things a neural network can consume.

`../motif-classifier` established the shape of this result and it is the prior we start
from: **the database is a good prior and a poor model.** Blending a learned pitch histogram
toward a DB-derived one at lambda=0.3 beat both the purely learned model (lambda=0) and the
purely prescriptive one (lambda=1), for both the histogram (M12) and the bigram LM (M13).
Nothing here should be used *instead of* training data; everything here is a prior on it.

Four views of the DB, in increasing order of how much structure they impose:

    affinity_matrix()      raag-to-raag similarity. Used for graded label smoothing (P1) --
                           the target says "Bageshree, and Bheempalasi would not be crazy".
    swar_occupancy()       (50, 12) -- how much of a raag's phrase material sits on each
                           swar. An auxiliary regression target (P2).
    pitch_template()       (50, n_bins) -- the same thing spread onto a continuous cents
                           grid, directly comparable to a model's pooled CQT (P3).
    scale_mask()           (50, 12) 0/1 -- which swars are legal at all. The bluntest prior.

All of it is read straight out of utils/raagdb.py and utils/raagspace.py rather than
reimplemented, so this project and that one cannot drift apart on what the DB says.
"""

from functools import lru_cache

import numpy as np

from .data import labels as dataset_labels
from .paths import add_sibling_paths

EPS = 1e-12


@lru_cache(maxsize=1)
def _raags():
    """{folder: Raag} in *our* label order, with the order asserted to match."""
    from utils.raagdb import dataset_raags

    ours = dataset_labels()
    db = dataset_raags(names=ours)   # our labels are the class list; do not go looking
    missing = [r for r in ours if r not in db]
    if missing:
        raise KeyError(f"libmogra DB has no entry for {missing}")
    return [db[r] for r in ours]


@lru_cache(maxsize=1)
def affinity_matrix():
    """(A, A_rot, best_k) in our label order.

    A[i, j] is how musically close raag j is to raag i -- phrase n-gram TF-IDF cosine,
    scale Jaccard and thaat, combined. A_rot is the same after allowing the predicted
    raag's scale to be rotated, which separates "wrong raag" from "right melody, wrong Sa".
    """
    from utils.raagspace import affinity

    lab, A, A_rot, best_k = affinity()
    ours = dataset_labels()
    if list(lab) != list(ours):
        idx = [list(lab).index(r) for r in ours]
        A, A_rot, best_k = A[np.ix_(idx, idx)], A_rot[np.ix_(idx, idx)], best_k[np.ix_(idx, idx)]
    return A, A_rot, best_k


def soft_targets(gamma=4.0):
    """(50, 50) row-stochastic. `q[i] ∝ affinity(i, ·) ** gamma` -- musically graded label
    smoothing: mass concentrated on the true raag, the remainder distributed over raags a
    listener could actually confuse it with, instead of uniformly over 49 strangers.

    `gamma` sets the peak: 1 is very soft, 8 is nearly one-hot. 4 is the value
    ../motif-classifier/musical_eval.py uses for its `affinity_ce`, so training against
    these targets and scoring with that metric use the same notion of "close".
    """
    A, _, _ = affinity_matrix()
    q = np.asarray(A, dtype=np.float64) ** gamma
    return (q / np.maximum(q.sum(axis=1, keepdims=True), EPS)).astype(np.float32)


@lru_cache(maxsize=1)
def swar_occupancy():
    """(50, 12) row-stochastic: how often each swar appears across a raag's mukhyanga
    phrases plus its aaroha and avaroha.

    This is the field that separates the pairs vaadi/samvaadi cannot. Bageshree and
    Bheempalasi share scale, vaadi *and* samvaadi, but differ here at L1 0.43 -- and in the
    musically correct direction (Bageshree weakens P and leans on D).
    """
    out = np.zeros((len(dataset_labels()), 12), dtype=np.float64)
    for i, r in enumerate(_raags()):
        for seq in list(r.phrases) + [r.aaroha, r.avaroha]:
            for s in seq:
                out[i, s] += 1.0
    return (out / np.maximum(out.sum(axis=1, keepdims=True), EPS)).astype(np.float32)


@lru_cache(maxsize=1)
def scale_mask():
    """(50, 12) 0/1 -- the swars a raag may legitimately use."""
    out = np.zeros((len(dataset_labels()), 12), dtype=np.float32)
    for i, r in enumerate(_raags()):
        for s in r.scale:
            out[i, s] = 1.0
    return out


@lru_cache(maxsize=8)
def pitch_template(n_bins=144, sigma_cents=35.0):
    """(50, n_bins) row-stochastic: the occupancy above, spread onto a continuous
    octave-folded cents grid with a Gaussian.

    `n_bins=144` matches this project's CQT (36 bins/octave x 4 octaves, folded), so a
    model's octave-folded CQT energy and this template live in the same vector space and
    can be compared bin-for-bin. Delegates to motif-classifier's `db_histogram` so the
    template is literally the one M12 scored 0.400 with.
    """
    add_sibling_paths()          # motif-classifier is not a shared util; reach in explicitly
    from methods.m12_dbhist import db_histogram

    return np.stack([db_histogram(r, n_bins=n_bins, sigma_cents=sigma_cents)
                     for r in _raags()]).astype(np.float32)


if __name__ == "__main__":
    L = dataset_labels()
    A, A_rot, _ = affinity_matrix()
    occ, mask, T = swar_occupancy(), scale_mask(), pitch_template()
    print(f"{len(L)} raags | affinity {A.shape} | occupancy {occ.shape} | template {T.shape}")
    print(f"affinity diagonal all 1: {np.allclose(np.diag(A), 1.0)}")
    i, j = L.index("Bageshree"), L.index("Bheempalasi")
    print(f"Bageshree vs Bheempalasi: affinity {A[i, j]:.3f}, "
          f"same scale {np.array_equal(mask[i], mask[j])}, "
          f"occupancy L1 {np.abs(occ[i] - occ[j]).sum():.3f}")
    q = soft_targets()
    print(f"soft target for Bageshree: true mass {q[i, i]:.3f}; top neighbours "
          f"{[(L[k], round(float(q[i, k]), 3)) for k in np.argsort(-q[i])[1:5]]}")
