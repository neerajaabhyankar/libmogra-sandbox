"""Praat autocorrelation pitch tracker (via parselmouth) -> relative pitch trajectory.

Classic autocorrelation-based pitch tracker used widely in voice/ethnomusicology
pitch research. Voicing is native: Praat reports 0 Hz for unvoiced frames.
"""

import sys
from pathlib import Path

import numpy as np
import librosa
import parselmouth

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import run_pipeline  # noqa: E402
from visualize import plot_relative_pitch  # noqa: E402

TARGET_SR = 22050
TIME_STEP = 0.01  # -> 100 fps
PITCH_FLOOR = librosa.note_to_hz("C2")  # ~65 Hz
PITCH_CEILING = librosa.note_to_hz("C7")  # ~2093 Hz


def extract_relative_pitch_praat(audio: np.ndarray, sr: int, plot: bool = True,
                                  use_chroma: bool = False):
    """audio: mono float32 waveform. Returns list[note_segmentation.Note]."""
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=TARGET_SR)
    pitch = sound.to_pitch(time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)

    f0_hz = pitch.selected_array["frequency"]  # 0.0 where unvoiced
    voiced_mask = f0_hz > 0

    notes, tonic_hz = run_pipeline(f0_hz, voiced_mask, TIME_STEP)

    if plot:
        plot_relative_pitch(notes, title="Praat", use_chroma=use_chroma, tonic_hz=tonic_hz)

    return notes
