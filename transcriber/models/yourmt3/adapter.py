"""YourMT3+ (Chang et al., 2024): multi-instrument note events with instrument labels -> lead track.

Vendored code and checkpoint: `vendor/` (from the Hugging Face space mimbres/YourMT3, GPL-3.0).
The model runs in `worker.py`, a separate process, because its package names collide with ours.
It writes notes per instrument (MIDI program; 100 = singing voice) with integer MIDI pitches --
no pitch bends, so the track is semitone-quantised to A440.

Lead: the singing-voice notes if there are any; otherwise the pitched (non-drum) program with the
most note-time. Overlapping lead notes: the later onset wins. The notes are then held over
their durations on a 10 ms grid (contract.notes_to_track).
"""

import atexit
import json
import os
import subprocess
import sys

import numpy as np

from ... import config as C
from ...contract import Note, notes_to_track

NAME = "yourmt3"
P = C.YOURMT3
SR = 16000
_WORKERS = {}


def _worker(checkpoint):
    if checkpoint not in _WORKERS:
        w = subprocess.Popen([sys.executable, "-m", "transcriber.models.yourmt3.worker", checkpoint],
                             cwd=P["vendor"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             env={**os.environ, "PYTHONPATH": str(C.HERE.parent)})
        assert w.stdout.readline().strip() == b"ready", "YourMT3 worker failed to start"
        atexit.register(w.terminate)
        _WORKERS[checkpoint] = w
    return _WORKERS[checkpoint]


def raw_notes(audio, checkpoint=NAME):
    """[(is_drum, program, onset, offset, pitch)] for everything the model hears."""
    w = _worker(checkpoint)
    x = np.ascontiguousarray(audio, np.float32)
    w.stdin.write(len(x).to_bytes(8, "little") + x.tobytes())
    w.stdin.flush()
    return json.loads(w.stdout.readline())


def lead(raw):
    pitched = [n for n in raw if not n[0]]
    voice = [n for n in pitched if n[1] in P["voice_programs"]]
    if not voice and pitched:
        time = {}
        for n in pitched:
            time[n[1]] = time.get(n[1], 0.0) + n[3] - n[2]
        voice = [n for n in pitched if n[1] == max(time, key=time.get)]
    return [Note(n[2], n[3], float(n[4]), instrument=str(n[1])) for n in voice]


def transcribe(audio, sr, checkpoint=NAME):
    assert sr == SR, f"YourMT3 expects {SR} Hz"
    return notes_to_track(lead(raw_notes(audio, checkpoint)), len(audio) / SR, P["hop_s"])
