"""CREPE (torchcrepe) frame-level pitch tracker -> relative pitch trajectory.

Deep-net monophonic pitch tracker; voicing decided by a confidence threshold.
"""

import sys
from pathlib import Path

import numpy as np
import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import run_pipeline  # noqa: E402
from visualize import plot_relative_pitch  # noqa: E402

TARGET_SR = 16000
HOP_LENGTH = 160  # 10ms @ 16kHz -> 100 fps
CONFIDENCE_THRESHOLD = 0.4
MODEL_SIZE = "tiny"


def extract_relative_pitch_crepe(audio: np.ndarray, sr: int, plot: bool = True,
                                 provided_tonic_hz: float = None):
    """audio: mono float32 waveform. Returns list[note_segmentation.Note]."""
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    wav = torch.from_numpy(audio).float().unsqueeze(0)
    with torch.no_grad():
        f0_hz, confidence = torchcrepe_predict(wav)

    f0_hz = f0_hz.squeeze(0).numpy()
    confidence = confidence.squeeze(0).numpy()
    voiced_mask = confidence >= CONFIDENCE_THRESHOLD
    hop_seconds = HOP_LENGTH / TARGET_SR

    notes, tonic_hz = run_pipeline(f0_hz, voiced_mask, hop_seconds, provided_tonic_hz=provided_tonic_hz)

    if plot:
        plot_relative_pitch(notes, title="CREPE", tonic_hz=tonic_hz)

    return notes


def torchcrepe_predict(wav, device="cpu"):
    """`device` is a pure speed knob: "mps" was verified bit-identical to "cpu" on this
    corpus (0.000 cents difference, identical voicing) at ~3.8x the throughput."""
    import torchcrepe
    return torchcrepe.predict(
        wav,
        TARGET_SR,
        hop_length=HOP_LENGTH,
        fmin=50.0,
        fmax=2000.0,
        model=MODEL_SIZE,
        return_periodicity=True,
        batch_size=512,
        device=device,
        decoder=torchcrepe.decode.weighted_argmax,
    )
