"""pYIN (librosa) frame-level pitch tracker -> relative pitch trajectory.

Classic probabilistic-YIN melody tracker; voicing is the algorithm's own
HMM/Viterbi voiced_flag, not a confidence threshold we pick ourselves.
"""

import sys
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import run_pipeline  # noqa: E402
from visualize import plot_relative_pitch  # noqa: E402

TARGET_SR = 22050
FMIN = librosa.note_to_hz("C2")  # ~65 Hz
FMAX = librosa.note_to_hz("C7")  # ~2093 Hz
FRAME_LENGTH = 2048
HOP_LENGTH = FRAME_LENGTH // 4


def extract_relative_pitch_pyin(audio: np.ndarray, sr: int, plot: bool = True,
                                 use_chroma: bool = False):
    """audio: mono float32 waveform. Returns list[note_segmentation.Note]."""
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    f0_hz, voiced_flag, _voiced_prob = librosa.pyin(
        audio,
        fmin=FMIN,
        fmax=FMAX,
        sr=TARGET_SR,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        fill_na=0.0,
    )
    voiced_mask = voiced_flag & (f0_hz > 0)
    hop_seconds = HOP_LENGTH / TARGET_SR

    notes, tonic_hz = run_pipeline(f0_hz, voiced_mask, hop_seconds)

    if plot:
        plot_relative_pitch(notes, title="pYIN", use_chroma=use_chroma, tonic_hz=tonic_hz)

    return notes
