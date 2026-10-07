"""What every model adapter hands back: a lead-melody pitch track.

    Track          f0_hz (0 = unvoiced), hop_s, confidence -- the same shape Melodia gives
    Note           (t0, t1, midi pitch, confidence, instrument) -- what note-event models give
    notes_to_track hold each note's pitch over its duration
    resample       a track onto another hop (nearest frame)

An adapter is a module in models/<name>/adapter.py exposing NAME, SR and
`transcribe(audio, sr) -> Track`. Nothing here knows about raags, Sa or pakad-matcher.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    f0_hz: np.ndarray            # Hz, 0 where unvoiced
    hop_s: float                 # seconds per frame
    confidence: np.ndarray = None

    @property
    def seconds(self):
        return len(self.f0_hz) * self.hop_s


@dataclass(frozen=True)
class Note:
    t0: float
    t1: float
    midi: float                  # fractional MIDI pitch (69 = A4 = 440 Hz)
    confidence: float = 1.0
    instrument: str = ""


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def notes_to_track(notes, seconds, hop_s):
    """A track that holds each note's pitch from t0 to t1; later notes overwrite overlaps."""
    f0 = np.zeros(int(np.ceil(seconds / hop_s)), np.float32)
    for n in sorted(notes, key=lambda n: n.t0):
        f0[int(round(n.t0 / hop_s)):int(round(n.t1 / hop_s))] = midi_to_hz(n.midi)
    return Track(f0, hop_s)


def resample(track, hop_s, n=None):
    """`track` on a grid of `hop_s` (n frames; default: same duration), by nearest frame."""
    n = int(round(track.seconds / hop_s)) if n is None else n
    idx = np.minimum((np.arange(n) * hop_s / track.hop_s).round().astype(int), len(track.f0_hz) - 1)
    conf = None if track.confidence is None else track.confidence[idx]
    return Track(track.f0_hz[idx] if len(track.f0_hz) else np.zeros(n, np.float32), hop_s, conf)
