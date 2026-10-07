"""Basic Pitch (Spotify, ICASSP 2022), ONNX backend: its pitch-salience map -> a lead track.

The model's three outputs per frame (hop 256 samples at 22050 Hz, ~86 frames/s) are note,
onset and *contour*: a salience for each of 264 pitch bins, 3 per semitone from 27.5 Hz. It is
polyphonic and has no instrument labels, so the lead is chosen here: the path through the
contour map within [fmin, fmax] that maximises salience minus a cost per bin of jump (dynamic
programming), refined between bins by a parabola through the peak and its neighbours, and
corrected by the model's measured one-bin sharpness (config `bin_offset`); frames where the
path's salience is below `voiced_salience` are unvoiced. Windowing and unwrapping are
Basic Pitch's own (`basic_pitch.inference`); only the file reading is replaced, to take arrays.
"""

import numpy as np

from ... import config as C
from ...contract import Track

NAME = "basic_pitch"
P = C.BASIC_PITCH
SR = 22050
_MODEL = []


def _model():
    if not _MODEL:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model
        _MODEL.append(Model(build_icassp_2022_model_path(FilenameSuffix.onnx)))
    return _MODEL[0]


def maps(audio):
    """Basic Pitch's three maps for `audio` (22050 Hz), frame hop 256 samples:
    {"contour": (frames, 264), "note": (frames, 88), "onset": (frames, 88)}."""
    from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
    from basic_pitch.inference import unwrap_output, window_audio_file
    n_olap = 30                                          # as basic_pitch.inference.run_inference
    olap = n_olap * FFT_HOP
    x = np.concatenate([np.zeros(olap // 2, np.float32), np.asarray(audio, np.float32)])
    wins = np.stack([w for w, _ in window_audio_file(x, AUDIO_N_SAMPLES - olap)])
    out = {}
    for i in range(0, len(wins), P["batch"]):
        for k, v in _model().predict(wins[i:i + P["batch"]]).items():
            out.setdefault(k, []).append(v)
    return {k: unwrap_output(np.concatenate(v), len(audio), n_olap) for k, v in out.items()}


def salience(audio):
    """The contour map alone: (frames, 264)."""
    return maps(audio)["contour"]


def lead_path(sal, bins_hz):
    """Bin index per frame of the best continuous path through `sal` (frames, bins)."""
    from basic_pitch.constants import CONTOURS_BINS_PER_SEMITONE  # noqa: F401  (3, documents the cap)
    keep = np.flatnonzero((bins_hz >= P["fmin_hz"]) & (bins_hz <= P["fmax_hz"]))
    e = np.log(sal[:, keep] + 1e-4)
    k = np.arange(len(keep))
    trans = -P["jump_cost"] * np.minimum(np.abs(k[:, None] - k[None, :]), P["jump_cap_bins"])
    score, back = e[0].copy(), np.zeros(e.shape, int)
    for t in range(1, len(e)):
        cand = score[:, None] + trans                    # from (rows) -> to (cols)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], k] + e[t]
    path = np.zeros(len(e), int)
    path[-1] = int(np.argmax(score))
    for t in range(len(e) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return keep[path]


def transcribe(audio, sr):
    from basic_pitch.constants import FFT_HOP, FREQ_BINS_CONTOURS
    assert sr == SR, f"Basic Pitch expects {SR} Hz"
    sal = salience(audio)
    if sal is None or not len(sal):
        return Track(np.zeros(0, np.float32), FFT_HOP / SR)
    b = lead_path(sal, FREQ_BINS_CONTOURS)
    t = np.arange(len(b))
    y0, y1, y2 = (sal[t, np.clip(b + d, 0, sal.shape[1] - 1)] for d in (-1, 0, 1))
    shift = np.clip(0.5 * (y0 - y2) / np.where(y0 - 2 * y1 + y2 == 0, -1e-9, y0 - 2 * y1 + y2), -0.5, 0.5)
    hz = FREQ_BINS_CONTOURS[0] * 2.0 ** ((b + shift + P["bin_offset"]) / 36.0)
    return Track(np.where(y1 >= P["voiced_salience"], hz, 0.0).astype(np.float32), FFT_HOP / SR, y1)
