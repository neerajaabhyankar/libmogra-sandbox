# transcriber -- other ways to hear the lead melody

**Goal.** Replace or improve the first step of `../pakad-matcher` -- Melodia's pitch track -- with
newer transcription models, and see whether the downstream tasks get better (phrase matching,
aarohi/avarohi, nyas). Our audio: one lead (voice, or a solo instrument) over a tanpura drone, a
harmonium-like melodic accompaniment and tabla. Only the lead matters.

**Rules** (`CLAUDE.md`): imports go one way, `pakad-matcher` → `transcriber`; one subfolder per
model; ask before downloads and long runs; choose on training/validation, test once.

## Glossary

| term | meaning |
|---|---|
| **lead** | the main melodic line: the singer, or the solo instrument. Not tanpura, harmonium, tabla |
| **track** | a pitch value (Hz, 0 = silent/unvoiced) every `hop` seconds: what Melodia gives |
| **note events** | (start, end, pitch) triples, often with an instrument label: what MT3-style models give |
| **adapter** | the code that turns a model's own output into a lead **track** (`contract.py`) |
| **source** | which pitch track `pakad-matcher` runs on: `melodia` (today) or a model here |
| **segments** | the time ranges `pakad-matcher` needs for choosing (notated chunks, validation spans, training/validation insight clips), padded with context |

## Design

```
transcriber/                         (this folder; imports nothing from pakad-matcher)
  contract.py       Track, Note; notes -> track; the adapter interface
  cache.py          run a model over (audio path, segments) -> cache/<model>/<key>.npz
  models/<name>/    adapter.py + README.md (outputs, adaptation, cost), one per model

pakad-matcher/
  config.py         PAKAD_PITCH_SOURCE=<model> switches the pitch track and the results folder
  fullaudio.py      contour() reads the chosen source; every script downstream is unchanged
  transcribers/
    segments.py     the time ranges each source must cover (no test ranges until the final pick)
    run.py          transcribe segments -> fit reader on notation -> phrase val -> insight val
    <model>/results/  that source's reader.json, s7_choice.json, insights/ ... (same files as results/)
```

Why a switch rather than new pipelines: the reader fit, phrase matcher, test2 and insights all
read pitch through `fullaudio.contour()`. Swapping what it returns reuses every line of them, and
each source's numbers land in its own folder, next to the Melodia ones in `results/`.

**Note-event models** become tracks by holding each lead note's pitch over its duration (as S13
did with the reader's notes); microtonal detail is lost, which is part of what gets measured.

**Per model, the same steps** (all on training/validation):
1. Adapter: model output → lead track. Pick the lead (instrument label, or melody selection from
   a polyphonic output); optional separation front-end (`../raag-identifier/source-separation`).
2. Reader refit on notation with this source (`fit_reader.py`) → misread rate (held out by recording).
3. Phrase matching on validation (`s7.py --val`), insights on train+validation (`insights.evaluate --val`).
4. Compare with Melodia on the same numbers. Only the final pick is scored on test, once.
5. If promising: adapt (fine-tune / low-rank adaptation) on the notation's per-note times
   (`annotations/adjusted_notations.jsonl`, training split only; recordings never straddle).

## Candidates

| model | what it outputs | lead / multi-track | singing in training? | cost here (M1, 16 GB) | status |
|---|---|---|---|---|---|
| **CREPE** (`torchcrepe`, installed) | track, 10 ms, 20-cent bins + confidence | monophonic: follows the loudest pitch | yes (incl. MIR-1K, MedleyDB) | ~17 s / 30 s on MPS; no download | 🟨 adapter ready; full run skipped for now (Neeraja: CREPE was tried elsewhere) |
| **Basic Pitch** (Spotify, 2022, Apache-2.0) | 3 maps at ~86 frames/s: pitch salience (3 bins per semitone ≈ 33 c), note, onset → note events with pitch bends | polyphonic, no instrument labels: the lead must be selected | partly (vocal datasets in its mix) | ~1 s / 30 s on CPU | 🔄 installed; adapter + calibration done; pakad-matcher run going |
| **YourMT3+** (2024, GPL-3.0, PyTorch Lightning) | note events with instrument labels, incl. a **singing voice** class | multi-track by design: take the voice/lead program | yes (vocal datasets) | ~8.6 s / 30 s on MPS, 22 s load; 562 MB checkpoint | 🔄 vendored; adapter (worker process) done; transcribing |
| **MT3** (Google, 2022, Apache-2.0, JAX/T5X) | note events with instrument labels | multi-track | **no** singing in its training sets | needs git-HEAD flax, seqio, t5x, note-seq + TensorFlow-pinned tensorflow-text: would re-pin the shared env's TensorFlow/numpy | 🟥 blocked (2026-10-07). Stand-in: YourMT3's "YMT3+" checkpoint (an MT3-architecture model, trained with voice) |
| **Transcription with Transformers** (Magenta, 2021) | piano note events | piano only | no | as MT3 | 🟥 skip: piano-only; MT3 is its multi-instrument extension |

Worth adding beyond the list (to discuss): **RMVPE** (vocal pitch straight from a mixture, built
for singing over accompaniment) and a CREPE/Basic Pitch fine-tune on our notation.

## Adaptation: probe the inner layers, then fine-tune (2026-10-07)

**Why.** Off the shelf, every model is worse than Melodia downstream (`../pakad-matcher/transcribers/compare.md`).
Their *final* outputs are the problem as much as the models: semitone-snapped notes, piano-style
note decisions. Their inner representations may still carry what we need.

**Labels.** Neeraja's notation with per-note times (`../pakad-matcher/annotations/adjusted_notations.jsonl`,
training split only; 43 recordings, 3,527 notes). Per 10 ms frame: the notated swar's pitch from Sa
(equal temperament, with octave; tracker-independent, so no model is favoured) on note frames;
"silent" on frames of a notated chunk outside every notated stretch; ignored elsewhere.

**Head.** A small network over per-frame feature blocks → a score per 20-cent pitch bin
(50–1600 Hz) plus "silent" (`adapt/`). Decoded to a track: best bin, refined by its neighbours.

**Steps.**
0. 🟥 **Fix label octave first (found 2026-10-07):** in 21 of 44 notated recordings Melodia sits an
   octave (20) or two (1) above the notation's octave -- the annotated Sa is likely an octave low
   for those voices. Plan: per recording, shift the label octave by the mode of (Melodia − notation)
   over notes where they agree in pitch class (8% of such notes disagree with their recording's
   mode). Until then, use pitch-class accuracy only. Baselines (pitch-class acc / voicing recall /
   false voicing): Melodia 0.795 / 0.942 / 0.394; basic_pitch 0.619 / 0.756 / 0.682.
1. 🟨 **Probes, backbones frozen** (CPU; code ready: `features.py`, `adapt.py`,
   `../pakad-matcher/transcribers/probe.py`; not yet run): blocks = spectrogram, Basic Pitch maps, Melodia track,
   and combinations. Scored held out by recording (4 folds) on the frame metrics below.
2. 🟥 **YourMT3+ / YMT3+ encoder states as a block** (GPU, after the Demucs queue).
3. 🟥 **Low-rank adaptation (LoRA)** of the best encoder with the head, if step 2 shows signal.
4. 🟥 **Downstream:** the best probes become pitch sources (`probe_<blocks>`) for the pakad-matcher
   pipeline. Notated recordings get predictions from the fold model that never saw them.

**Frame metrics** (held out by recording; on notation frames):
- *pitch accuracy* = share of note frames where the output is voiced and within 50 c of the label (= the right swar and octave);
- *pitch-class accuracy* = the same, with octave errors forgiven;
- *voicing recall* = share of note frames output as voiced;
- *false voicing* = share of "silent" frames output as voiced.

## Steps

- ✅ Folder, `CLAUDE.md`, this plan, model survey (2026-10-07)
- ✅ Infrastructure: `contract.py`, `cache.py`, `config.py`; the `pakad-matcher` switch, `segments.py`, `run.py` (smoke-tested)
- 🟨 CREPE end to end (no download): adapter works (`models/crepe/README.md`); full run (~45 min transcription + reader refit) waiting for Neeraja's go
- ✅ Installs via poetry (2026-10-07): `basic-pitch[onnx]` (macOS marker: plain `poetry add` would have downgraded TensorFlow 2.17 → 2.11 through basic-pitch's Linux-only pins), `lightning`, `einops`, `deprecated`. YourMT3+ code + checkpoint vendored (538 MB). HT-Demucs weights were already cached.
- ✅ Separation front-end: source `<model>+<backend>` (e.g. `basic_pitch+demucs`) in `cache.py`, reusing `../raag-identifier/source-separation`
- ✅ Basic Pitch end to end (2026-10-07): worse than Melodia on every validation number (`../pakad-matcher/transcribers/compare.md`); fewer voiced frames → many deleted notes and choppy pauses. Next: tune `voiced_salience` / `jump_cost` on notation
- 🔄 YourMT3+ transcription of the choosing ranges (~4–5 h: decoding slows with dense accompaniment notes), then end to end
- 🔄 MT3 stand-in `ymt3plus` (Neeraja 2026-10-07: skip MT3 -- no singing in its training); checkpoint 518 MB, ~28 s / 30 s
- 🔄 Separation variants (Neeraja: run them): `basic_pitch+demucs`, `yourmt3+demucs`, `ymt3plus+demucs`
- Queue: GPU transcriptions one at a time; each source's reader/phrase/insight steps start when its transcription ends
- 🟥 Adaptation on notation

## Log

- **2026-10-07** -- folder set up; survey above (sources: the four papers/repos Neeraja listed).
  Infrastructure + CREPE adapter; one notated range: CREPE vs Melodia median 10 c apart where both voiced.
