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
| **CREPE** (`torchcrepe`, installed) | track, 10 ms, 20-cent bins + confidence | monophonic: follows the loudest pitch | yes (incl. MIR-1K, MedleyDB) | fast on MPS; no download | 🟨 control: a learned monophonic tracker |
| **Basic Pitch** (Spotify, 2022, Apache-2.0) | 3 maps at ~86 frames/s: pitch salience (3 bins per semitone ≈ 33 c), note, onset → note events with pitch bends | polyphonic, no instrument labels: the lead must be selected | partly (vocal datasets in its mix) | tiny model, fast on CPU; pip package | 🟥 needs install |
| **YourMT3+** (2024, GPL-3.0, PyTorch Lightning) | note events with instrument labels, incl. a **singing voice** class | multi-track by design: take the voice/lead program | yes (vocal datasets) | large transformer; CPU or MPS (untested), slow; checkpoints on Hugging Face | 🟥 needs code + checkpoint |
| **MT3** (Google, 2022, Apache-2.0, JAX/T5X) | note events with instrument labels | multi-track | **no** singing in its training sets | JAX/T5X on M1 is fragile; Colab-oriented | 🟥 likely superseded by YourMT3+ (its PyTorch successor, with voice) |
| **Transcription with Transformers** (Magenta, 2021) | piano note events | piano only | no | as MT3 | 🟥 skip: piano-only; MT3 is its multi-instrument extension |

Worth adding beyond the list (to discuss): **RMVPE** (vocal pitch straight from a mixture, built
for singing over accompaniment) and a CREPE/Basic Pitch fine-tune on our notation.

## Steps

- ✅ Folder, `CLAUDE.md`, this plan, model survey (2026-10-07)
- ✅ Infrastructure: `contract.py`, `cache.py`, `config.py`; the `pakad-matcher` switch, `segments.py`, `run.py` (smoke-tested)
- 🟨 CREPE end to end (no download): adapter works (`models/crepe/README.md`); full run (~45 min transcription + reader refit) waiting for Neeraja's go
- 🟥 Basic Pitch, YourMT3+, separation front-end: waiting on download approval
- 🟥 Adaptation on notation

## Log

- **2026-10-07** -- folder set up; survey above (sources: the four papers/repos Neeraja listed).
  Infrastructure + CREPE adapter; one notated range: CREPE vs Melodia median 10 c apart where both voiced.
