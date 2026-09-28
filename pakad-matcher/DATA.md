# The data, the words, and the rules

Two kinds of human work feed this project, and they must not be mixed up. This file explains both,
defines every term that appears in `plan.md` and the code, and states the split rules.

**`audit.py` is the enforcing version of this document.** It computes the splits, prints the
current inventory, and fails loudly if a rule is broken. When this file and `audit.py` disagree,
`audit.py` is right and this file needs fixing.

```bash
poetry run python audit.py
```

---

## Glossary

Grouped by topic, not alphabetically; the terms only make sense next to each other. Written for
someone who knows ML and Hindustani music but not this project. **If a term in `plan.md`, a
figure or a docstring is not here, that is a bug in this file.**

### The music

| term | meaning |
|---|---|
| **swar letters** | `S r R g G m M P d D n N`: lower case = komal (flat) for r g d n, **M = teevra (sharp) ma**, m = shuddha ma. Same letters as libmogra |
| **saptak marks** | octave. `,n` = mandra (below Sa), `n` = madhya (middle), `` `n `` = taar (above) |
| **tonic** (Sa) | the reference pitch of a recording, in Hz, annotated by hand in `tonics.csv`. Never estimated here |
| **samooha** (phrase, pakad) | a short characteristic swar sequence of a raag, e.g. `m D n D`. Ours are in `neeraja_mukhyangas.json`; libmogra's list is its *mukhyanga* |
| **samooha id** | `Raag#3` = index 3 of that raag's list in libmogra's `RAAG_DB` (0-indexed); `Raag#N1` = one of Neeraja's own |
| **aaroh / avaroh** | the ascending / descending movement of a raag |
| **aarohi / avarohi swar** | **X is avarohi if only notes *below* X may come after X; aarohi if only notes *above* X may. What comes before X does not matter.** (Vrindavani Sarang: `m P n P N S R n P` is valid: n → P, N → S.) |
| **unidirectional swar** | aarohi or avarohi, as opposed to **bidirectional** (either may follow) |
| **kan** | a grace note: a brief touch of a neighbouring swar |
| **meend** | a glide between swars; the swars it passes through are not "sung" |
| **alap / madhya / taan** | slow and unmetred / medium tempo / fast runs. As *chunk kinds* they mean what the model measures (below), not what a musician would call them |

### The audio

| term | meaning |
|---|---|
| **recording** | one full performance, e.g. `8ldBWSCfR0Q`, named by YouTube id. Everything is split by recording, never inside one |
| **clip** | a 20 s excerpt in the pinned HuggingFace dataset. Early stages only; the annotation work uses full recordings |
| **contour** (pitch track) | one number per frame, in **cents above Sa** (100 cents = one semitone), or *unvoiced* (no pitch). From Essentia's Melodia, which quantises to 10 cents |
| **frame, hop** | one contour sample; the hop between frames is ≈ 18 ms (Melodia's 4.4 ms, downsampled ×4) |
| **held frames / held note** | frames where pitch moves slower than `held_slope` (800 cents/s, measured over 90 ms) for at least 0.1 s: the voice is *sitting* on a note rather than gliding. A **held note** is one such run. Pure pitch-track geometry, no swar knowledge |
| **density** (notes per second) | held notes per second over a window: the model's measure of tempo. Chooses alap (lowest), madhya (median), taan (highest) chunks, and the reader's onset cost |

### What a musician contributes

| term | meaning |
|---|---|
| **candidate** | a span of audio the matcher proposes as one occurrence of a samooha: recording + start + end |
| **judgment** | **one y/n on one candidate**: "is this really that samooha?". Made in the phrase app (`/phrases`), stored in `annotations/labels.jsonl`. Blank = unsure, never scored |
| **pool** | the fixed candidates offered for one samooha (14, sometimes 12). A judgment is stored as *an index into its pool*, so a pool is never rebuilt once judged — only **extended** by appending (`pool.py --extend`) |
| **tempo spreading** | pools interleave slow, medium and fast candidates rather than taking only the cheapest, which are usually quick transits |
| **chunk** | a 15–20 s stretch of a recording chosen for notating (`annotations/chunks.json`) |
| **stretch** | a notated sub-range of a chunk: times, the swars heard, and how they were placed |
| **notation** | the swars a musician heard in a stretch. `annotations/notations.jsonl`. **Training data** |
| **aligned / spaced by hand** | how a stretch's swars were placed in time: `align` fits them to the contour; `even` spreads them evenly because the tracker lost the voice. Hand-spaced = evidence of *what*, not *when* |
| **round** | a batch of samoohas or chunks added together (`round` in the json, `*_R3`/`*_R4` in `config.py`); `plan.md` says what each added |

### The split

| term | meaning |
|---|---|
| **train** | notation. Everything the models learn from musical ears |
| **seen / unseen raag** | seen = at least one notated stretch; unseen = none (`config.UNNOTATED_RAAGS`) |
| **validation** | judgments in seen raags, plus those in `config.VALIDATION_RAAGS` (unseen raags held out for choosing). Used to **choose** a method, never to grade one |
| **test1** | judgments in unseen raags: "is this the samooha?" |
| **test2** | `neeraja_unidirectionals.json`: for 24 swars in 6 raags, "used in aaroh? used in avaroh?" — 48 y/n questions |
| **control** (test2) | a bidirectional swar included so a method that calls everything unidirectional is caught; two per raag |
| **R1–R6** | the split rules, below; `audit.py` enforces them |
| **frozen choice** | the method picked on validation is written to `results/s7_choice.json` (with its hash) *before* test is scored |
| **wrong-tonic recording** | a recording whose `tonics.csv` Sa is wrong; listed in `config.BAD_TONIC_VIDEOS` and left out of training |
| **contaminated** | fitted on data that is now test. `config.MATCH`'s tuned values (S4b) are: they were fitted on the first 168 judgments |

### The models

| term | meaning |
|---|---|
| **matcher** | `matcher.match`: given a contour and a samooha, finds the spans that best fit it. A left-to-right model: one state per swar (each held for at least `min_dwell_s`), **ornament** states between them for kan and meend, free start and end. Returns candidates with a **cost** |
| **cost** | how badly a span fits a samooha; lower is better. Not comparable across samoohas |
| **matcher constants** | `config.MATCH`. `free_cents`: how far off a swar pitch may be for free; `scale_cents`, `note_cap`: how the penalty grows beyond that; `orn_cost`, `transit_cost`: price of ornament frames; `note_trim`: fraction of a note's frames that must fit; `leap_penalty`: a step going the wrong way or octave; `register_penalty`: sung in a different saptak than written |
| **alignment** | placing a *given* swar sequence onto a contour (`decode.align`). Used by the notation app, and to check notation |
| **reader** / **reading** | the swar sequence the model hears in a contour **with no samooha to guide it** (`decode.free_read`). The same note and ornament states as the matcher, but any swar may follow any other |
| **onset cost** | what the reader pays to start a new note. Too low and every glide becomes several notes (the reader's main failure: it over-segments) |
| **swar centres / offsets** | where each swar actually sits, in cents from equal temperament |
| **what the reader learned** | **from notation only** (`fit_reader.py` → `results/reader.json`): the onset cost, separately for slow and fast stretches (by density), and the 12 swar offsets (which come out small). Its other constants are hand-set (`config.NOTATE_MATCH`). Its held-out misread rate is 0.58 |
| **free edges / rim** | an alignment may leave silence or drone at the ends of a selection unexplained |

### Methods compared (S7–S10)

| term | meaning |
|---|---|
| **hand-set** | the matcher with constants chosen by hand, before any judgment existed. The **baseline** |
| **S4b-tuned** | `config.MATCH`: constants tuned on the first 168 judgments. Reference only (contaminated) |
| **notation-set** | hand-set, with tolerance and dwell taken from the notation, and the reader's swar offsets |
| **read-then-match** | the reader transcribes the span; the score is how few edits turn the samooha into some stretch of that transcription |
| **combined** | notation-set cost + a weight × read-then-match; the weight chosen on validation |
| **val-tuned** | hand-set constants re-tuned on validation judgments (coordinate ascent on per-samooha AUC) |
| **leave-one-samooha-out** | how a method tuned on validation is scored *on* validation: tune on all samoohas but one, score that one, repeat. Otherwise it grades itself |

### Test2 methods and definitions

| term | meaning |
|---|---|
| **departure** | each occurrence of swar X counts as up or down by the **next** note — the definition of aarohi/avarohi above |
| **up-fraction** | of X's occurrences, the fraction followed by a higher note. The "used in aaroh?" score; 1 − it is the "used in avaroh?" score. The x-axis of `results/roc/test2_scatter.png` |
| **held-notes only (untuned)** | occurrences = held notes (above) snapped to the nearest swar of the raag's scale. **Nothing fitted to notation**; its one threshold (`held_slope`) was tuned in S4b on judgments |
| **tuned heuristic notes** | occurrences = the reader's notes, restricted to the raag's scale: the reader with its onset cost and swar centres fitted to notation |
| **phrase** (in `unidir.py`) | a voiced stretch between silences longer than 0.35 s; direction is never judged across a silence |

### Metrics

| term | meaning |
|---|---|
| **misread rate** | `(substitutions + deletions + insertions) / notated swars` between the reader and the notation, like word error rate in speech. 0 is perfect |
| **per-samooha AUC** | the chance that a "yes" candidate outscores a "no" *of the same samooha*, averaged over samoohas. 0.5 is chance. Samoohas that came back all-yes or all-no have none |
| **P@1, P@3** | of the 1 or 3 candidates ranked highest for a samooha, the fraction judged "yes". Averaged over samoohas |
| **ROC curve** | true-positive rate against false-positive rate as the threshold moves; area under it = AUC. In `results/roc/`. For test1, scores are first converted to their **rank within their samooha**, since raw costs do not compare across samoohas |
| **test2 AUC** | pooled over all 48 questions (up-fractions do compare across swars) |
| **paired difference, CI** | method A − method B per samooha, averaged; the 95% interval is by bootstrap over samoohas. "10 better / 3 worse of 15" counts samoohas |

---

## The files

| path | what | written by |
|---|---|---|
| `neeraja_mukhyangas.json` | the samoohas, hand-picked, with saptak marks and provenance | by hand |
| `annotations/labels.jsonl` | **judgments** — one line per y/n, append-only, last line wins | the phrase app |
| `annotations/pool/*.json` | the candidates offered per samooha. **Frozen once judged** | `pool.py` |
| `annotations/chunks.json` | the chunks offered for notating | `chunks.py` |
| `annotations/notations.jsonl` | **notations** — one line per save of a chunk, last wins | the notation app |
| `annotations/audio/`, `annotations/chunks/` | the wav snippets both apps play | `pool.py`, `chunks.py` |
| `cache/f0_essentia_full.npz` | pitch tracks and salience for whole recordings | `fullaudio.py` |
| `results/` | everything computed; nothing here is human input | the analysis scripts |

---

## The split, and why it is drawn this way

The test set answers the question the tool exists for: *hand it a samooha, does it find real
occurrences?* So the test set is **judgments**, and nothing may be fitted on them.

The training set is **notations**, because they teach the general thing — how a musician's ear
segments a contour into swars — without ever being the question we score.

Two contaminations are possible, and both are handled by recording *and* by time:

| rule | |
|---|---|
| **R1** | a judgment in a raag with **no notation** is **test** (changed 2026-09-27; it used to be "a recording with no notation") — the test asks whether the tool works on raags it never learned from |
| **R2** | a judgment in a **notated raag**, not overlapping a notated stretch (5 s of margin), is **validation** — seen raags help choose methods, never grade them |
| **R3** | a judgment overlapping a notated stretch is **unusable**: neither fitted on nor scored |
| **R4** | raags in `config.UNNOTATED_RAAGS` are never notated, so all their judgments are test — except R6 |
| **R5** | a pool is never rebuilt once it carries judgments — labels are indices into it, so regenerating one silently re-points them. `pool.py` refuses without `--force` |
| **R6** | judgments in `config.VALIDATION_RAAGS` (un-notated raags, fixed before judging) are **validation**. Without them, validation holds only raags the reader learned, and S7 showed that picks the wrong method |

As of 2026-09-27: **test 232** judgments over 17 samoohas in 9 un-notated raags; **validation 207**
over 15 samoohas (12 in notated raags, 3 in Alhaiya Bilawal and Tilang); 1 set aside.

**Guards in code, not just in prose:**

- `pool.py` refuses to rebuild an existing pool (R5).
- `chunks.py` skips `UNNOTATED_RAAGS` and any recording that already carries a judgment (R4, and
  keeps R1 growing rather than shrinking). It also **adds** chunks rather than rewriting
  `chunks.json`: notations refer to chunks by id, and a chunk's `video`/`t0` is what turns a
  stretch's times into recording times, so rewriting one orphans the work done on it.
- `audit.py` recomputes R1–R6 from the files and fails if any is broken.

**Consequence worth remembering:** 42 recordings now carry judgments, which leaves almost nothing
free in the six originally notated raags. Further notation therefore comes from **fresh raags**
(round 3: Yaman, Bhairav, Malkauns, Bhoopali, Jog, Kalawati). The reader's parameters are
raag-independent, so this is not a compromise — it tests whether they generalise.

---

## Where it stands (2026-09-27; regenerate with `audit.py`)

| | |
|---|---|
| train, notation | 47 chunks · 230 stretches · **2291 swars** · 628 s · 12 raags · 28 recordings |
| validation, judgments | **207** over 15 samoohas (12 in seen raags, 3 in Alhaiya Bilawal and Tilang) |
| test1, judgments | **232** over 17 samoohas in 9 unseen raags |
| test2 | 48 questions over 24 swars in 6 raags |
| set aside (R3) | 1 |
| awaiting judgment | 20 appended candidates (Kedar#2, Marwa#1) |
| awaiting notation | **37 chunks** — Charukeshi, Hindol, Ahir Bhairav, Durga (round 4), and 13 madhya chunks |

Anything fitted before 2026-09-24 — the tuned matcher costs in `config.MATCH` and the probability
in `results/calibration.json` — was fitted on what is now the test set, and is marked as such in
`plan.md`. Those numbers do not count until they are re-derived from training data.
