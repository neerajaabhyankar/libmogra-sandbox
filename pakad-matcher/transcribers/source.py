"""A ../transcriber model's cached tracks, laid on Melodia's frame grid for fullaudio.contour().

The only place pakad-matcher imports the transcriber package (imports go one way). Frames
outside the transcribed ranges are unvoiced, so segments.py must cover everything a run reads.
"""

import sys

import numpy as np

import config as C

if str(C.HERE.parent) not in sys.path:
    sys.path.append(str(C.HERE.parent))          # appended: pakad-matcher's own modules win

from transcriber import cache, contract  # noqa: E402


def on_grid(video, n, hop, model=None):
    """f0 in Hz over `n` frames of `hop` s (0 = unvoiced), from the model's cached ranges."""
    out = np.zeros(n, np.float32)
    for t0, tr in cache.load(model or C.PITCH_SOURCE, video):
        a = int(round(t0 / hop))
        f0 = contract.resample(tr, hop).f0_hz[:max(0, n - a)]
        out[a:a + len(f0)] = f0
    return out
