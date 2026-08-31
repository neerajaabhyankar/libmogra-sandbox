"""A raag-to-raag affinity matrix, so "how wrong is this mistake" has a number.

Top-1 accuracy treats every error identically: calling Tilak Kamod "Des" (same thaat, same
S R G m P D N material, genuinely confusable by ear) scores exactly as badly as calling it
"Bairagi" (a five-note Bhairav-family raag with komal r, nothing in common). That is the
wrong scoring for this problem — a phrase matcher that lands in the right neighbourhood is
doing something right, and we want to see it.

The affinity is built **only from the libmogra database**, never from data or predictions,
so nothing here is fitted and it can be applied to any method's output. Three views of
"related", combined:

| component | what it captures |
|---|---|
| `phrase` | TF-IDF cosine over mukhyanga + aaroha/avaroha n-grams, following `../../raagspace.ipynb` — shared *melodic motion* |
| `scale`  | Jaccard over the swar sets — shared *material* |
| `thaat`  | same parent scale family — shared *lineage* |

and separately, the thing that is not a musical mistake at all:

| `rotation` | the best affinity obtainable by rotating the predicted raag's scale by k semitones |

A confusion with high rotational affinity but low direct affinity means the model heard the
melody correctly and put Sa in the wrong place. That is a **tonic error**, not a raag error,
and it is worth counting separately — it points at the pitch pipeline, not the matcher.
"""

from functools import lru_cache

import numpy as np

from raagdb import SWAR_NAMES, collapse, dataset_raags


def _docs(raags, order):
    """One whitespace document of swar tokens per raag — the raagspace.ipynb encoding,
    extended with aaroha/avaroha since the tuning showed those carry as much signal."""
    docs = []
    for folder in order:
        r = raags[folder]
        parts = []
        for seq in list(r.phrases) + [r.aaroha, r.avaroha]:
            seq = collapse(seq)
            if len(seq) >= 2:
                parts.append(" ".join(SWAR_NAMES[s] for s in seq))
        docs.append(" . ".join(parts))
    return docs


def _rotate_scale(scale, k):
    return {(s + k) % 12 for s in scale}


def _jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


@lru_cache(maxsize=4)
def affinity(w_phrase=0.4, w_scale=0.45, w_thaat=0.15):
    """Returns (labels, A, A_rot, best_k).

    A       (R,R) in [0,1], 1 on the diagonal — direct musical affinity.
    A_rot   (R,R) in [0,1] — best affinity over the 12 rotations of the predicted raag.
    best_k  (R,R) int — which rotation achieved it (0 means no rotation needed).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    raags = dataset_raags()
    order = sorted(raags)
    R = len(order)

    vec = TfidfVectorizer(analyzer="word", ngram_range=(2, 4), lowercase=False,
                          token_pattern=r"\S+")
    X = vec.fit_transform(_docs(raags, order)).toarray()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    phrase = X @ X.T

    scales = [raags[f].scale for f in order]
    scale = np.array([[_jaccard(a, b) for b in scales] for a in scales])

    thaats = [(raags[f].thaat or "").lower() for f in order]
    thaat = np.array([[1.0 if a and a == b else 0.0 for b in thaats] for a in thaats])

    A = w_phrase * phrase + w_scale * scale + w_thaat * thaat
    A = A / (w_phrase + w_scale + w_thaat)
    np.fill_diagonal(A, 1.0)

    # rotational view: scale-only, since the phrase TF-IDF has no rotated counterpart
    A_rot = np.zeros((R, R))
    best_k = np.zeros((R, R), dtype=int)
    for i in range(R):
        for j in range(R):
            vals = [_jaccard(scales[i], _rotate_scale(scales[j], k)) for k in range(12)]
            best_k[i, j] = int(np.argmax(vals))
            A_rot[i, j] = float(np.max(vals))
    return order, A, A_rot, best_k


def neighbours(folder, n=6):
    """The n musically closest raags to `folder` — a sanity check on the matrix."""
    order, A, _, _ = affinity()
    i = order.index(folder)
    idx = np.argsort(-A[i])
    return [(order[j], round(float(A[i, j]), 3)) for j in idx if j != i][:n]


if __name__ == "__main__":
    order, A, A_rot, best_k = affinity()
    print(f"{len(order)} raags; mean off-diagonal affinity "
          f"{(A.sum() - np.trace(A)) / (A.size - len(order)):.3f}")
    for f in ("TilakKamod", "Bhoopali", "Bageshree", "Todi", "Yaman"):
        print(f"  {f:14s} -> {neighbours(f, 5)}")
    print("\n  a deliberately bad pair, for contrast:")
    i, j = order.index("TilakKamod"), order.index("Bairagi")
    print(f"  TilakKamod vs Bairagi: direct {A[i, j]:.3f}, best-rotation {A_rot[i, j]:.3f} at k={best_k[i, j]}")
