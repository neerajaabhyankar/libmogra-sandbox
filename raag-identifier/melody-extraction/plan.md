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
  -> histogram(note_segments)                              # the same notes, as a distribution
       duration-weighted. Two views, drawn as a pair, and they are the two ends of a
       scale rather than the same thing twice:
         left  — raw cents, 10-cent bins, octaves kept apart, nothing snapped. An
                 intonation sitting 30 cents off its semitone shows up 30 cents off,
                 which is the reason to look at a histogram in this repo at all.
         right — chroma: folded to one octave *and* quantized to the nearest semitone,
                 so twelve bars, one per swara. Throws away exactly what the left panel
                 is for, and what survives — the swara set and how time divides between
                 those twelve — is the part that compares across clips.
       `main_live` shows both under the trajectory panels, against the Sa just hummed.
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
                         # draw_relative_pitch_multi(axes, ...) for callers owning the figure
  freq_histogram.py      # plot_pitch_histogram(notes_or_cents) -> pitch distribution
                         # plot_audio_histogram(audio_or_path) -> same, from raw audio
                         # plot_relative_pitch_with_histograms(...) -> what main_live shows
                         # also runnable: python freq_histogram.py clip.wav --octave-wrap
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

## Notes on the histograms

- The octave-folded window is **[-50, 1150) cents, not [0, 1200)**. With a [0, 1200)
  fold, a Sa sung 20 cents flat lands at 1180 and the Sa peak splits across the two
  ends of the axis — the single most-visited pitch in the clip becoming two half-height
  peaks at opposite edges. Shifting the window by half a semitone puts every semitone
  class in the middle of its own stretch of axis, so each peak stays whole.
- Quantizing to chroma is **snap first, fold second**. Folding first would send a Sa
  sung 20 cents flat to 1180 cents and then round it into N's bin; snapping first sends
  it to 1200, which folds cleanly onto S. (The [-50, 1150) window makes the two steps
  agree anyway — 100-cent bins on that window are already centred on the semitones —
  but the order is what keeps it true if the window ever moves.)
- The histograms are built from **note events, duration-weighted**, not from raw frames,
  because that is what the trackers return and what `main_live` already has in hand (no
  re-running a tracker to redraw). The cost is resolution: `segment_notes` already
  averaged each note over a ±50-cent band, so the result is spikier than a true
  frame-level pitch distribution. If shruti-level peak *shape* ever matters more than
  peak *location*, the fix is to have the trackers also hand back frame-level cents —
  `pitch_histogram()` already accepts a bare cents array, so only the trackers change.

## Out of scope (intentionally)

- No training, no classifiers, no saved embeddings/checkpoints.
- No raag-level statistics or hand-crafted features — that's the user's next step.
- No Melodia/Essentia for now (install friction on macOS arm64) — revisit only if the
  four current methods prove insufficient. (Note that the Vamp plugin route Tony uses
  would also work for Melodia, which ships as a Vamp plugin.)
