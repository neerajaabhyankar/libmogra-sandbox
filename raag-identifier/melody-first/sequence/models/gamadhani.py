# GaMaDhaNi-inspired melody parser.
#
# GaMaDhaNi (Shikarpur et al., ISMIR 2024) is a generative model for Hindustani
# melodic vocal contours that operates on a finely-quantized, tonic-normalized
# pitch contour. We use *only their pitch representation scheme*, not the
# generative model — the output here is the pitch contour itself as a
# (T, 1) sequence, which is exactly the "sequence embedding" for Phase 2.
#
# Pipeline:
#   audio -> CREPE pitch tracker (torchcrepe) at 100 fps
#         -> voiced frames only (confidence threshold)
#         -> estimate tonic = mode of voiced pitch distribution (rounded to
#            nearest semitone, i.e. 100-cent grid)
#         -> convert all frames to cents relative to tonic
#         -> silence/unvoiced frames set to SILENCE_VALUE
#   output: (T, 1) float32, values in approx [-1200, 1200] cents (±1 octave)
#           SILENCE_VALUE for unvoiced frames.
#
# Augmentation for Phase 2b: add a uniform random offset in cents to the entire
# contour (excluding silence frames) — this shifts the tonic and makes the model
# tonic-agnostic without any tonic labels.

import numpy as np
import torch
import librosa

from .base import BaseSeqEmbedder

TARGET_SR = 16000        # torchcrepe's native rate
CREPE_HOP_LENGTH = 160   # 10ms per frame at 16kHz → 100 fps
CONFIDENCE_THRESHOLD = 0.4
SILENCE_VALUE = 0.0      # cent value used for unvoiced frames


def _estimate_tonic_hz(f0_hz: np.ndarray, confidence: np.ndarray) -> float:
    """Mode of the voiced pitch distribution, snapped to nearest semitone."""
    voiced = f0_hz[confidence >= CONFIDENCE_THRESHOLD]
    voiced = voiced[voiced > 0]
    if len(voiced) == 0:
        return 440.0  # fallback
    cents = 1200.0 * np.log2(voiced / 440.0)
    # round each frame to nearest 100 cents (semitone) then find mode
    semitones = np.round(cents / 100.0) * 100.0
    unique, counts = np.unique(semitones, return_counts=True)
    tonic_cents = unique[np.argmax(counts)]
    return 440.0 * 2.0 ** (tonic_cents / 1200.0)


class GaMaDhaNiEmbedder(BaseSeqEmbedder):
    name = "gamadhani"

    def load(self):
        import torchcrepe
        self._torchcrepe = torchcrepe

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        if sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)

        wav = torch.from_numpy(audio_array).float().unsqueeze(0)  # (1, T)

        with torch.no_grad():
            f0_hz, confidence = self._torchcrepe.predict(
                wav,
                TARGET_SR,
                hop_length=CREPE_HOP_LENGTH,
                fmin=50.0,
                fmax=2000.0,
                model="tiny",
                return_periodicity=True,
                batch_size=512,
                device="cpu",
                decoder=self._torchcrepe.decode.weighted_argmax,
            )

        f0_hz = f0_hz.squeeze(0).numpy()         # (T,)
        confidence = confidence.squeeze(0).numpy()  # (T,)

        # Tonic estimation from voiced frames
        tonic_hz = _estimate_tonic_hz(f0_hz, confidence)

        # Convert to cents relative to tonic; silence unvoiced frames
        with np.errstate(divide="ignore", invalid="ignore"):
            cents = np.where(
                (confidence >= CONFIDENCE_THRESHOLD) & (f0_hz > 0),
                1200.0 * np.log2(np.clip(f0_hz, 1e-6, None) / tonic_hz),
                SILENCE_VALUE,
            )

        return cents.astype(np.float32).reshape(-1, 1)  # (T, 1)
