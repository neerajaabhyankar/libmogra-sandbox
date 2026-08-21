"""Tony (Sonic Visualiser) melody transcription -> relative pitch trajectory.

Tony <https://www.sonicvisualiser.org/tony/> is a GUI app, but its transcription
engine is the pYIN Vamp plugin (Mauch & Dixon 2014) -- the GUI is a hand-correction
front-end over it. So instead of driving the app, we host the same plugin in-process
via the `vamp` Python module and read the two outputs Tony's two panes display:

    smoothedpitchtrack  frame-level f0 after pYIN's Viterbi pitch smoothing
    notes               Tony's note-level HMM transcription (start, duration, Hz)

That second output is what makes this different from `pyin_tracker.py`: librosa's
pyin gives frame f0 only, and we then segment notes ourselves. Tony has its own
note HMM (onset sensitivity + duration pruning) doing the segmentation instead.

Needs the pyin Vamp plugin installed -- run `python trackers/tony/install_pyin_plugin.py`.
"""

import os
import sys
import contextlib
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline import estimate_tonic_hz, run_pipeline  # noqa: E402
from note_segmentation import Note  # noqa: E402
from visualize import plot_relative_pitch  # noqa: E402

PLUGIN_KEY = "pyin:pyin"
TARGET_SR = 44100  # pYIN's block/step sizes are frame counts; Tony's usual rate
STEP_SIZE = 256  # plugin's preferred step -> ~5.8 ms hop at 44.1 kHz

# pYIN parameter defaults, as Tony ships them.
#   outputunvoiced=2 ("yes, as negative frequencies") is ours, not Tony's: it is the
#   only setting under which smoothedpitchtrack emits one value per frame, so the
#   returned vector stays time-aligned. With the default (0) unvoiced frames are
#   dropped and the remaining values silently slide earlier in time.
TONY_PARAMETERS = {
    "threshdistr": 2.0,        # Beta (mean 0.15)
    "outputunvoiced": 2.0,
    "precisetime": 0.0,
    "lowampsuppression": 0.1,
    "onsetsensitivity": 0.7,   # note HMM: how eagerly it splits a slide into notes
    "prunethresh": 0.1,        # note HMM: drop notes shorter than this (max 0.2)
}


@contextlib.contextmanager
def _quiet_native_output():
    """The pyin plugin prints `mnOut size: ...` from C++; swallow it.

    Has to happen at the file-descriptor level -- the writes come from the dylib,
    so redirecting sys.stdout/sys.stderr in Python would not see them.
    """
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        for fd in saved:
            os.close(fd)


def _run_pyin(audio, sr, parameters):
    """Returns (f0_hz[T], voiced_mask[T], hop_seconds, tony_notes).

    tony_notes is the plugin's own note list: [{t_start, t_end, f0_hz}, ...].
    """
    try:
        import vamp
    except ImportError as e:
        raise ImportError(
            "The `vamp` module is missing. Install it with:\n"
            "    pip install --no-build-isolation vamp\n"
            "(build isolation must be off: its setup.py imports numpy directly.)"
        ) from e

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    audio = np.asarray(audio, dtype=np.float32)

    with _quiet_native_output():
        track = vamp.collect(
            audio, TARGET_SR, PLUGIN_KEY,
            output="smoothedpitchtrack", parameters=parameters, step_size=STEP_SIZE,
        )
        note_out = vamp.collect(
            audio, TARGET_SR, PLUGIN_KEY,
            output="notes", parameters=parameters, step_size=STEP_SIZE,
        )

    hop_seconds, values = track["vector"]
    f0_signed = np.asarray(values, dtype=np.float64)
    voiced_mask = f0_signed > 0  # unvoiced frames come back as negative frequencies
    f0_hz = np.abs(f0_signed)

    tony_notes = [
        {
            "t_start": float(f["timestamp"]),
            "t_end": float(f["timestamp"]) + float(f["duration"]),
            "f0_hz": float(f["values"][0]),
        }
        for f in note_out["list"]
    ]
    return f0_hz, voiced_mask, float(hop_seconds), tony_notes


def extract_relative_pitch_tony(audio: np.ndarray, sr: int, plot: bool = True,
                                provided_tonic_hz: float = None,
                                note_source: str = "tony",
                                parameters: dict = None):
    """audio: mono float32 waveform. Returns list[note_segmentation.Note].

    note_source="tony"     notes come from the pYIN note HMM, i.e. what Tony draws
                           in its note pane. Duration pruning is the plugin's own
                           `prunethresh` (default 0.1 s, raise to 0.2 to match the
                           0.2 s floor the other trackers use).
    note_source="pipeline" notes come from our shared segment_notes() instead, so
                           the result is directly comparable to pyin/crepe/praat --
                           only the frame-level f0 differs.
    """
    params = dict(TONY_PARAMETERS)
    if parameters:
        params.update(parameters)

    f0_hz, voiced_mask, hop_seconds, tony_notes = _run_pyin(audio, sr, params)

    if note_source == "pipeline":
        notes, tonic_hz = run_pipeline(f0_hz, voiced_mask, hop_seconds,
                                       provided_tonic_hz=provided_tonic_hz)
    elif note_source == "tony":
        # Tonic still comes from the frame-level track, exactly as in run_pipeline,
        # so the cents reference is identical across both modes and all trackers.
        tonic_hz = (provided_tonic_hz if provided_tonic_hz is not None
                    else estimate_tonic_hz(f0_hz, voiced_mask))
        notes = [
            Note(t_start=n["t_start"], t_end=n["t_end"],
                 cents_relative=1200.0 * np.log2(n["f0_hz"] / tonic_hz))
            for n in tony_notes if n["f0_hz"] > 0
        ]
    else:
        raise ValueError(f"note_source must be 'tony' or 'pipeline', got {note_source!r}")

    if plot:
        label = "note HMM" if note_source == "tony" else "shared pipeline"
        plot_relative_pitch(notes, title=f"Tony ({label})", tonic_hz=tonic_hz)

    return notes
