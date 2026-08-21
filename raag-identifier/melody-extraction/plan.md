# melody-extraction: relative-pitch transcription

Sibling to `melody-first`, scoped much narrower. `melody-first` explored methods that
extract *track-level* or *sequence* embeddings for similarity/classification (CREPE-based
contours, CQT cover-detection nets, generative pitch models). None of those, except the
pitch-tracking step buried inside GaMaDhaNi, actually output a usable note-level pitch
transcription — they output embeddings, not pitch.

This folder does one thing: **audio in → voiced-only, tonic-normalized relative-pitch
trajectory out**, as a plain function call you run ad hoc (no batch extraction, no
persistence, no model training). Anything past this point (hand-crafted raag features,
statistics, etc.) is out of scope here — that's the user's next step, by hand.

## What's reused from `melody-first` (no recreation)

- `melody-first/sequence/models/gamadhani.py::_estimate_tonic_hz` — the tonic heuristic
  (mode of the voiced-pitch distribution, snapped to the nearest semitone). Imported
  directly, not reimplemented.

That's the only piece of `melody-first` that's actually suitable here. Everything else in
that folder (CoverHunter, Discogs-VINet, bytecover, re-move, clews) is a cover-song /
track-fingerprint embedding model — they consume CQT/audio and emit a fixed-size vector
for similarity search. They don't expose a pitch curve at all, so there's nothing to reuse
from them for *note transcription*.

## Methods chosen for frame-level pitch tracking

Four monophonic pitch trackers, picked for being genuinely different algorithms (deep
net / classical probabilistic / classical autocorrelation) with different failure modes,
each with a real voicing decision (not just "is there energy"):

| Method | Library | Voicing detection | Notes |
|---|---|---|---|
| **CREPE** | `torchcrepe` (already used in melody-first) | confidence threshold | deep net, robust to timbre/noise, slower, needs the `tiny`/`full` checkpoint |
| **pYIN** | `librosa.pyin` | native (HMM/Viterbi `voiced_flag`) | classic MIR melody-transcription algorithm, CPU-only, fast |
| **Praat autocorrelation** | `parselmouth` (Praat) | native (returns 0 Hz when unvoiced) | classic in voice/ethnomusicology pitch research, different smoothing behavior than YIN-family methods, CPU-only |
| **Tony** | pYIN Vamp plugin via `vamp` | native (unvoiced frames as negative Hz) | the transcription engine behind the Tony GUI; adds pYIN's **note-level HMM**, which `librosa.pyin` does not implement — see `trackers/tony/README.md` |

Plain deterministic YIN was considered and dropped — pYIN is its probabilistic superset
with built-in voicing, so it adds no new failure mode worth comparing. Essentia/Melodia
(the CompMusic-standard predominant-melody extractor) was considered but has no working
macOS arm64 wheel without a from-source build; flagged below as a possible future add
rather than burning time on a fragile install.

## Shared pipeline (one implementation, four frame-level inputs)

```
audio, sr
  -> [method-specific] frame-level f0_hz[T], voiced_mask[T], hop_seconds
  -> tonic_hz = estimate_tonic(f0_hz, voiced_mask)      # reused heuristic
  -> cents[T] = 1200 * log2(f0_hz / tonic_hz), NaN where unvoiced
  -> note_segments = segment_notes(cents, voiced_mask, hop_seconds, min_note_dur=0.2s)
       merges consecutive voiced frames that stay within a cents tolerance band into
       one note event {t_start, t_end, cents_relative}; this is the literal "if you must
       quantize, use 0.2s" rule — notes shorter than that get absorbed into a neighbor.
  -> chroma_relative = cents_relative mod 1200             # octave-folded variant
  -> plot(note_segments)                                   # step plot, gaps at unvoiced
       one panel per tracker: raw cents solid, semitone-quantized behind it in the
       same colour at low alpha, so quantization error reads as a halo on the line.
       The y-axis auto-fits the notes but never scales past visualize.PLOT_RANGE_CENTS
       (G(-1)..D(+1)); octave-error strays outside it are cut off rather than allowed
       to squash the melody, and the panel says how many were dropped.
```

Implemented once in `pipeline.py` / `note_segmentation.py` / `visualize.py`, shared by all
four tracker modules. Tony is the one partial exception: by default its notes come from
pYIN's own note HMM rather than `segment_notes()`, since that HMM is the reason to use it
at all. Pass `note_source="pipeline"` for the strictly comparable version.

## Folder layout

```
melody-extraction/
  plan.md
  pipeline.py            # shared: tonic call, cents conversion, chroma fold
  note_segmentation.py   # frame f0 -> discrete note events (min 0.2s)
  visualize.py           # plot_relative_pitch(notes) -> matplotlib Figure (not saved)
                         # plot_relative_pitch_multi([(label, notes), ...]) stacks trackers
  trackers/
    crepe_tracker.py     # extract_relative_pitch_crepe(audio, sr, plot=True)
    pyin_tracker.py       # extract_relative_pitch_pyin(audio, sr, plot=True)
    praat_tracker.py      # extract_relative_pitch_praat(audio, sr, plot=True)
    tony/                # self-contained: needs a Vamp plugin binary, unlike the others
      README.md
      tony_tracker.py    # extract_relative_pitch_tony(audio, sr, plot=True)
      install_pyin_plugin.py
```

## API contract

Each tracker module exposes one function:

```python
def extract_relative_pitch_<method>(audio: np.ndarray, sr: int, plot: bool = True) -> list[dict]:
    """Returns note events: [{"t_start": float, "t_end": float,
                              "cents_relative": float, "chroma_cents": float}, ...]
    Unvoiced regions are simply absent (gaps), not zero-filled.
    If plot=True, shows a matplotlib plot of the trajectory; nothing is written to disk.
    """
```

Nothing is cached, batched, or written to `outputs/` — call it directly on whatever clip
you're looking at, in a notebook or script.

## Out of scope (intentionally)

- No training, no classifiers, no saved embeddings/checkpoints.
- No raag-level statistics or hand-crafted features — that's the user's next step.
- No Melodia/Essentia for now (install friction on macOS arm64) — revisit only if the
  four current methods prove insufficient. (Note that the Vamp plugin route Tony uses
  would also work for Melodia, which ships as a Vamp plugin.)
