"""Scoring methods. Each exposes `score(feat) -> np.ndarray` over the 50 candidate raags.

Shared contract:
  * `Method.raags` is the ordered list of candidate folder names.
  * `Method.score_at(feat, k)` scores a clip at one tonic rotation `k`.
  * `Method.score(feat)` handles the tonic search (`shift_mode`) and returns the final
    per-raag vector plus the rotation it settled on.
"""

import numpy as np


class Method:
    """Base: candidate list, tonic-rotation search, prediction."""

    name = "base"

    def __init__(self, raags, shift_mode="none", shifts=tuple(range(12))):
        self.raags = list(raags)
        self.shift_mode = shift_mode  # "none" | "global" | "per_raag"
        self.shifts = (0,) if shift_mode == "none" else tuple(shifts)

    #: methods that estimate anything from train data set this, so the CV harness knows to
    #: refit per fold instead of scoring with a model that has seen the held-out clips
    fitted = False

    def fit(self, feats):
        """Estimate whatever this method learns from train clips. Default: nothing."""
        return self

    def score_at(self, feat, k):
        raise NotImplementedError

    def score(self, feat):
        """Returns (scores over self.raags, chosen rotation).

        "global"    one rotation per clip, the one whose best raag scores highest —
                    commits to a single tonic reading of the clip.
        "per_raag"  each raag gets its own best rotation. More permissive (and more prone
                    to spurious matches), but it lets a raag be found even when the clip's
                    dominant pitch is not its Sa.
        """
        if self.shift_mode == "none":
            return self.score_at(feat, 0), 0
        stack = np.stack([self.score_at(feat, k) for k in self.shifts])  # (K, R)
        if self.shift_mode == "per_raag":
            return stack.max(axis=0), int(self.shifts[int(np.argmax(stack.max(axis=1)))])
        best_k = int(np.argmax(stack.max(axis=1)))
        return stack[best_k], int(self.shifts[best_k])

    def predict(self, feat):
        s, _ = self.score(feat)
        return self.raags[int(np.argmax(s))]
