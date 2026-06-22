"""Shared tonic estimation + relative-pitch conversion, used by all trackers.

Tonic estimation is imported (not reimplemented) from melody-first's GaMaDhaNi
wrapper: mode of the voiced-pitch distribution, snapped to the nearest semitone.
"""

import sys
from pathlib import Path

import numpy as np

_SEQUENCE_DIR = Path(__file__).resolve().parent.parent / "melody-first" / "sequence"
if str(_SEQUENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SEQUENCE_DIR))

from models.gamadhani import _estimate_tonic_hz  # noqa: E402

from note_segmentation import segment_notes  # noqa: E402


def estimate_tonic_hz(f0_hz: np.ndarray, voiced_mask: np.ndarray) -> float:
    """Reuses melody-first's heuristic: mode of voiced pitch, snapped to nearest semitone."""
    confidence = voiced_mask.astype(np.float32)  # _estimate_tonic_hz just thresholds at 0.4
    return _estimate_tonic_hz(f0_hz, confidence)


def run_pipeline(f0_hz, voiced_mask, hop_seconds, tol_cents=50.0, min_note_dur=0.2):
    """f0_hz, voiced_mask: equal-length 1D arrays at a fixed frame hop.

    Returns (notes, tonic_hz) where notes is a list of note_segmentation.Note.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    voiced_mask = np.asarray(voiced_mask, dtype=bool)

    tonic_hz = estimate_tonic_hz(f0_hz, voiced_mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.clip(f0_hz, 1e-6, None) / tonic_hz)

    notes = segment_notes(cents, voiced_mask, hop_seconds, tol_cents=tol_cents, min_note_dur=min_note_dur)
    return notes, tonic_hz
