# The data, the words, and the rules

Two kinds of human work feed this project, and they must not be mixed up. This file explains both,
defines every term that appears in `plan.md` and the code, and states the split rules.

**`audit.py` is the enforcing version of this document.** It computes the splits, prints the
current inventory, checks them against the frozen manifest, and fails loudly if a rule is broken.
When this file and `audit.py` disagree, `audit.py` is right and this file needs fixing.

```bash
poetry run python audit.py            # rules + inventory + manifest check
poetry run python audit.py --freeze   # pin the splits (results/splits_manifest.json)
```

**Owner's rules for inference (Neeraja, 2026-10-03):** the input is **audio plus its Sa** — never
the raag. Raag labels may be used to train or tune; they never restrict, prime or answer at
inference, and the raag DB (libmogra) is never an input. Restricting notes to the labelled raag's
scale during evaluation was tolerated earlier as a convenience; since 2026-10-03 nothing does it.

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
| **tonic** (Sa) | the reference pitch of a recording, in Hz. **Given at inference** (owner, 2026-10-03); here it comes from the hand-annotated `tonics.csv` and is never estimated |
| **samooha** (phrase, pakad) | a short swar sequence of a raag that someone wants to find, e.g. `m D n D`. Ours are in `neeraja_mukhyangas.json`; libmogra's list is its *mukhyanga*. Whether a samooha is *characteristic* of its raag is not scored here. ("Phrase" means only this; voiced stretches between breaths are **breath spans**) |
| **samooha id** | `Raag#3` = index 3 of that raag's list in libmogra's `RAAG_DB` (0-indexed); `Raag#N1` = one of Neeraja's own |
| **aaroh / avaroh** | the ascending / descending movement of a raag |
| **aarohi / avarohi swar** | **X is avarohi if only notes *below* X may come after X; aarohi if only notes *above* X may. What comes before X does not matter.** (Vrindavani Sarang: `m P n P N S R n P` is valid: n → P, N → S.) Exceptions are sung as artistic liberty (Jog G→g, Malhar N→n meends) |
| **next note** | what decides the direction of X: the next **sung** note after X — kan and pass-through notes are skipped — and never across a breath (owner, 2026-10-03). In code: `notes.directions`; kan = notes shorter than `config.NOTES.kan_max_s` (0.08 s); breath = an unvoiced run ≥ `breath_s` (0.25 s) |
| **unidirectional swar** | aarohi or avarohi, as opposed to **bidirectional** (either may follow) |
| **nyas** | the swar a breath or pause follows. **Not** the longest note ("a long Ga … ending on a short Re: Re is the nyas"), and **not** a phrase end (`P M G m G R S` in one breath rests on S). In fast passages the pause can be short, relative to the pace |
| **kan** | a grace note: a brief touch of a neighbouring swar |
| **meend** | a glide between swars; the swars it passes through are not "sung" |
| **andolan** | a slow oscillation on a swar |
| **audav** | a raag with five swars (e.g. Bhoopali, Durga) |
| **lay** | tempo. **madhya lay** = medium tempo |
| **alap / madhya / taan** | slow and unmetred / medium tempo / fast runs. As *chunk kinds* they mean what the model measures (by **density**), not what a musician would call them |

### The audio

| term | meaning |
|---|---|
| **recording** | one full performance, e.g. `8ldBWSCfR0Q`, named by YouTube id. Everything is split by recording, never inside one |
| **clip** | a 20 s excerpt in the pinned HuggingFace dataset (early stages only); an **insight clip** is a 30 s excerpt of a full recording (below) |
| **contour** (pitch track) | one number per frame, in **cents above Sa** (100 cents = one semitone), or *unvoiced* (no pitch). From Essentia's Melodia, which quantises to 10 cents |
| **frame, hop** | one contour sample; the hop between frames is ≈ 18 ms (Melodia's 4.4 ms, downsampled ×4) |
| **held frames / held note** | frames where pitch moves slower than `held_slope` (800 cents/s, measured over 90 ms) for at least 0.1 s: the voice is *sitting* on a note rather than gliding. Pure pitch-track geometry, no swar knowledge |
| **breath span** | a voiced stretch of contour between breaths (unvoiced runs ≥ `config.NOTES.breath_s`), at least `min_phrase_s` long (`notes.breath_spans`). Direction is never judged across one boundary |
| **transit** | frames between two notes that belong to neither (a glide), priced by `transit_cost` |
| **density** (notes per second) | held notes per second over a window: the model's measure of tempo. Chooses alap (lowest), madhya (median), taan (highest) chunks, and the reader's onset cost |
| **voice above drone** | `insights/voice.py`: per frame, spectral energy (150–4000 Hz) above each frequency's quiet-end level over the clip — the steady tanpura removed. A loudness number, not a spectrogram model |

### What a musician contributes

| term | meaning |
|---|---|
| **candidate** | a span of audio the matcher proposes as one occurrence of a samooha: recording + start + end |
| **judgment** | **one y/n on one candidate**: "is this really that samooha, *intended*?". **Yes** includes intentional ornaments; **no** usually means the notes were only passed through in a meend, caught stray, or an unintentional ornament (owner, 2026-09-27). Made in the phrase app (`/phrases`), stored in `annotations/labels.jsonl`. Blank = unsure, never scored |
| **pool** | the fixed candidates offered for one samooha: 14 (12 in a few), 24 after an **extension**. A judgment is stored as *an index into its pool*, so a pool is never rebuilt once judged — only extended by appending (`pool.py --extend`). Pools were shortlisted by the matcher as configured when they were built (`config.MATCH`, the S4b costs), so test1 only holds spans that matcher already ranked well |
| **tempo spreading** | pools interleave slow, medium and fast candidates rather than taking only the cheapest, which are usually quick transits |
| **chunk** | a 15–20 s stretch of a recording chosen for notating (`annotations/chunks.json`) |
| **stretch** | a notated sub-range of a chunk: times, the swars heard, and how they were placed |
| **notation** | the swars a musician heard in a stretch — **including notes the pitch track misses** (tanpura Sa on top, a tapering voice). `annotations/notations.jsonl`. **Training data** |
| **aligned / spaced by hand** | how a stretch's swars were placed in time: `align` fits them to the contour; `even` spreads them evenly because the tracker lost the voice. Hand-spaced = evidence of *what*, not *when* |
| **notation_notes / f0_agrees** | `annotations/notation_notes.jsonl` (`corpus.py --notes`): every notated note with its placement. `f0_agrees` = the pitch track is within 50 cents of the swar and mostly voiced; false = heard but not seen by the pitch track. Kept, for methods that don't rely on f0. Pitch-based fits still use every note (open) |
| **insight clip** | a 30 s madhya-lay stretch annotated for insights: per swar a **direction label** (aarohi / avarohi / both / not sung / unsure, *as sung in that clip*) and **nyas windows** (a stretch of pitch track and the swar resting there). `annotations/insight_clips.json`, labels in `annotations/insights.jsonl` |
| **eyeball set** | the 15 clips first shown to Neeraja *with the machine's findings* (2026-10-01), then labelled as the insight test set. Her labels came after seeing the machine output |
| **round** | a batch of samoohas or chunks added together (see Inventory) |

### The split

| term | meaning |
|---|---|
| **train** | notation (phrase task, reader); train-split insight clips (insights). Everything models learn from |
| **seen / unseen raag** | seen = at least one notated stretch; unseen = none (`config.UNNOTATED_RAAGS`) |
| **validation** | judgments in seen raags, plus those in `config.VALIDATION_RAAGS` (unseen raags held for choosing). Methods are **chosen** here, and may be **fitted** here (val-tuned) — but then they are scored on validation only leave-one-samooha-out |
| **test1** | judgments in unseen raags: "is this the samooha?" |
| **test2** | `neeraja_unidirectionals.json`: for 24 swars in 6 raags, "used in aaroh? used in avaroh?" — 48 y/n questions, but only **24 independent swars** (each swar answers both) |
| **control** (test2) | a bidirectional swar included so a method that calls everything unidirectional is caught; two per raag, picked by Claude |
| **insight splits** | insight clips are train (10) / validation (8) / test (15, one excluded): rules I-R1–I-R4 in `insights/clips.py` — no recording in two splits; test and validation never on a notated recording; a train clip never overlaps a notated stretch; the registry is append-only |
| **R1–R7** | the split rules, below; `audit.py` enforces them |
| **manifest** | `results/splits_manifest.json`: which label is in which split, with a hash, frozen *before* choosing (`audit.py --freeze`). Choices and test results record the hash they were made with |
| **frozen choice** | what was picked without test, written before test is scored: `results/s7_choice.json` (phrase method), `results/insights/choice.json` (+ detector numbers) |
| **test look** | one scoring of a test set. History per set is in `plan.md` § Review; since 2026-10-03 each test is scored once, after everything is frozen |
| **wrong-tonic recording** | a recording whose `tonics.csv` Sa is wrong (`config.BAD_TONIC_VIDEOS`); excluded **everywhere** (R7) |
| **S4b-tuned / "contaminated"** | `config.MATCH`'s "(tuned)" values were fitted on the first 168 judgments (S4b). Under today's split those are **validation** (167) and set aside (1) — none is test. So they are validation-fitted, not test-contaminated (earlier text said "test"; that was wrong) |

### The models

| term | meaning |
|---|---|
| **matcher** | `matcher.match`: given a contour and a samooha, finds the spans that best fit it. A left-to-right model: one state per swar (each held for at least `min_dwell_s`), **ornament** states between them for kan and meend, free start and end. Returns candidates with a **cost** |
| **cost** | how badly a span fits a samooha; lower is better. Not comparable across samoohas |
| **matcher constants** | `config.MATCH`. `free_cents`: how far off a swar pitch may be for free; `scale_cents`, `note_cap`: how the penalty grows beyond that; `orn_cost`, `transit_cost`: price of ornament frames; `note_trim`: fraction of a note's frames that must fit; `leap_penalty`: a step going the wrong way or octave; `register_penalty`: sung in a different saptak than written |
| **IoU** | intersection over union of two time spans; candidates overlapping more than `NMS_IOU` are suppressed |
| **alignment** | placing a *given* swar sequence onto a contour (`decode.align`). Used by the notation app, and to check notation |
| **reader** / **reading** | the swar sequence the model hears in a contour **with no samooha to guide it** (`decode.free_read`). The same note and ornament states as the matcher, but any swar may follow any other |
| **tuned heuristic notes** | the reader's notes with times (`notes.notes`), using the reader fitted to notation. What test2 and the insights count |
| **onset cost** | what the reader pays to start a new note, separately for slow and fast stretches (by density) |
| **swar centres / offsets** | where each swar actually sits, in cents from equal temperament |
| **what the reader learned** | from notation only (`fit_reader.py` → `results/reader.json`): onset costs, swar offsets, tolerance, ornament prices, minimum note length. Its held-out misread rate is 0.559. Two constants it did not refit came from S4b (`held_slope`, `note_cap`; S11's search kept their values). Since 2026-10-03 `reader.json` stores **every** constant the reader uses, so editing `config.MATCH` no longer changes it |
| **learned reader** | `archive/ctc_reader.py` (S12): a GRU reading swars frame by frame. Held-out misread 0.618; not used |
| **free edges / rim** | an alignment may leave silence or drone at the ends of a selection unexplained |
| **threshold heuristics** | the hand-written insight rules (`insights/core.py`, values in `config.INSIGHTS`, frozen in `results/insights/choice.json`): direction = ups ≥ `dir_ratio` × downs; pause = an unvoiced run ≥ `pause_min_s` and ≥ `pause_rel` × the local median note length, or ≥ `pause_abs_s`; nyas = the last note before a pause. Not raag rules |
| **learned detectors** | `insights/detect.py`: logistic models, audio only. Nyas: is a breath/pause after this note end? (cues: pitch-track gap, its ratio to the pace, voice-above-drone and overall loudness drop, note length, slope, Sa/Pa). Direction: aarohi / avarohi / both per swar of a clip (cues: up/down counts at three minimum note lengths, time on the swar) |
| **proxy direction label** | from notation, not from a judgment: in a notated chunk, a swar whose notated moves (≥ 4) go ≥ 85% up is aarohi, ≥ 85% down avarohi, 25–75% both; 75–85% is left out as ambiguous |

### Methods compared (phrase task, S7–S11)

| term | meaning |
|---|---|
| **hand-set** | the matcher with constants chosen by hand, before any judgment existed. The **baseline** |
| **S4b-tuned** | `config.MATCH`: constants tuned on the first 168 judgments (all now validation). Reference only |
| **notation-set** | hand-set, with tolerance and dwell taken from the notation, and the reader's swar offsets |
| **read-then-match** | the reader transcribes the span; the score is how few edits turn the samooha into some stretch of that transcription |
| **combined** | notation-set cost + a weight × read-then-match; the weight chosen on validation |
| **val-tuned** | hand-set constants re-tuned on validation judgments (coordinate ascent on per-samooha AUC). The tool (`pakad.py`) runs whichever method `s7_choice.json` holds |
| **leave-one-samooha-out** | how a method tuned on validation is scored *on* validation: tune on all samoohas but one, score that one, repeat |
| **leave-one-clip-out** | the same for insight variants, over train + validation clips; every variant (heuristics included) is refitted inside each fold |
| **held-notes only (untuned)** | test2 method: held notes snapped to the nearest of the 12 swars. "Untuned" = not fitted to notation; its held threshold is S4b's |
| **Platt scaling** | a logistic map from cost to probability (`calibrate.py`), fitted on validation under the chosen method |

### Metrics

| term | meaning |
|---|---|
| **misread rate** | `(substitutions + deletions + insertions) / notated swars` between the reader and the notation, like word error rate. 0 is perfect |
| **per-samooha AUC** | the chance that a "yes" candidate outscores a "no" *of the same samooha*, averaged over samoohas. 0.5 is chance. All-yes or all-no samoohas have none |
| **P@1, P@3** | of the 1 or 3 candidates ranked highest for a samooha, the share judged "yes", averaged over samoohas. **Ties at the cut are shared fairly** (expected value over tie orders) since 2026-10-03; before, pool order broke them |
| **ROC curve** | true-positive rate against false-positive rate as the threshold moves; area under it = AUC. In `results/roc/`. For test1, scores are first converted to their rank within their samooha |
| **test2 AUC** | pooled over the 48 questions (up-fractions compare across swars); its interval resamples the 24 swars |
| **Brier score** | mean squared error of a probability against 0/1 outcomes; lower is better |
| **balanced accuracy** | (insight directions) the mean of the recalls for aarohi, avarohi and both — so saying "both" for everything scores only 1/3 |
| **nyas F1** | each machine pause event (swar, time) is matched one-to-one to a marked nyas window if it starts within [window start − 0.3 s, window end + 0.6 s]; F1 of matched events whose swar (pitch class) is right. **Set F1**: the clip's nyas list against the swars marked |
| **paired difference, interval** | method A − method B on the same items; the 95% interval resamples whole units (samoohas, clips or swars) with replacement (`metrics.bootstrap`, 2000 resamples). Intervals reported before 2026-10-03 came from scripts that were not saved and cannot be reproduced |

### Stage names

`S1`–`S12` (phrase task and reader), `N1` (notation beyond f0), `I1`–`I6` (insights) and **Review**
are sections of `plan.md`, in the order the work was done.

---

## The files

| path | what | written by |
|---|---|---|
| `neeraja_mukhyangas.json` | the samoohas, hand-picked, with saptak marks and provenance | by hand |
| `neeraja_unidirectionals.json` | test2 ground truth | by hand |
| `annotations/labels.jsonl` | **judgments** — one line per y/n, append-only, last line wins | the phrase app |
| `annotations/pool/*.json` | the candidates offered per samooha. **Frozen once judged** | `pool.py` |
| `annotations/chunks.json` | the chunks offered for notating | `chunks.py` |
| `annotations/notations.jsonl` | **notations** — one line per save of a chunk, last wins | the notation app |
| `annotations/notation_notes.jsonl` | every notated note, placed, with `f0_agrees` | `corpus.py --notes` |
| `annotations/insight_clips.json`, `insights.jsonl` | insight clips and their labels | `insights/clips.py`, the insight app |
| `annotations/audio/`, `annotations/chunks/`, `annotations/insight_clips/` | the wav snippets the apps play (not committed) | `pool.py`, `chunks.py`, `insights/clips.py` |
| `cache/f0_essentia_full.npz` | pitch tracks and salience for whole recordings | `fullaudio.py` |
| `results/` | everything computed; nothing here is human input. Earlier versions of a result are kept with a stage suffix (`_r2`, `_s10`, `_s11`, `_i6`, …) | the analysis scripts |
| `archive/` | scripts of earlier stages, kept for provenance (`archive/README.md`) | — |

---

## The split, and why it is drawn this way

The test set answers the question the tool exists for: *hand it a samooha, does it find real
occurrences?* So the test set is **judgments**, and nothing may be fitted on them.

The training set is **notations**, because they teach the general thing — how a musician's ear
segments a contour into swars — without ever being the question we score.

| rule | |
|---|---|
| **R1** | a judgment in a raag with **no notation** is **test** (since 2026-09-27; before, "a recording with no notation") — the test asks whether the tool works on raags it never learned from |
| **R2** | a judgment in a **notated raag**, not overlapping a notated stretch (5 s of margin), is **validation**. (Neeraja's suggestion; such a judgment may share a recording with notation — 15 recordings do) |
| **R3** | a judgment overlapping a notated stretch is **unusable**: neither fitted on nor scored |
| **R4** | raags in `config.UNNOTATED_RAAGS` are never notated, so all their judgments are test — except R6 |
| **R5** | a pool is never rebuilt once it carries judgments — labels are indices into it. `pool.py` refuses without `--force` |
| **R6** | judgments in `config.VALIDATION_RAAGS` (un-notated raags) are **validation**. Added 2026-09-26, *after* the first test1 scoring (S7) showed validation without unseen raags picked the wrong method |
| **R7** | nothing on a wrong-tonic recording (`config.BAD_TONIC_VIDEOS`) is used anywhere — notation, judgments, test2 pooling, insight clips (2026-10-03) |
| test2 | pools every recording of its raags **except** notated ones (training) and wrong-tonic ones (since 2026-10-03; before, 3 notated Jog recordings were pooled) |

**Guards in code, not just in prose:**

- `audit.judgments()` is the only reader of judgments; everything that fits or scores goes through
  `audit.splits()`. (Archived S4/S4b/S5a scripts read the raw file — `archive/README.md` warns.)
- `audit.py --freeze` pins the splits; `s7.py`, `unidir.py` and `insights/evaluate.py` record and
  check the manifest hash.
- `pool.py` refuses to rebuild an existing pool (R5).
- `chunks.py` skips `UNNOTATED_RAAGS` and any recording that already carries a judgment, and only
  **adds** chunks: notations refer to chunks by id.

---

## Inventory (2026-10-03; regenerate with `audit.py`)

| | |
|---|---|
| train, notation | 82 chunks · 366 stretches · **3529 swars** · 1073 s · 16 raags · 44 recordings (1 wrong-tonic chunk left out) |
| validation, judgments | **207** over 15 samoohas (12 in seen raags, 3 in Alhaiya Bilawal and Tilang) |
| test1, judgments | **247** over 17 samoohas in unseen raags (5 more on the wrong-tonic Multani recording set aside) |
| set aside | 6 (1 overlapping notation, 5 wrong tonic) |
| test2 | 48 questions over 24 swars in 6 raags |
| insight clips | train 10 · validation 8 (7 labelled; Kaushik Dhwani saved empty) · test 14 (Multani excluded, wrong tonic) |

**Notation rounds** (which raags were notated when; formerly lists in `config.py`):
round 1–2 Bageshree, Darbari Kanada, Malhar, Puriya Dhanashri, Shree, Bheempalasi · round 3
(2026-09-24) Yaman, Bhairav, Malkauns, Bhoopali, Jog, Kalawati — chosen away from the test raags,
two of them audav · round 4 (2026-09-27) Charukeshi, Hindol, Ahir Bhairav, Durga, plus madhya-lay
chunks in Yaman, Bhairav, Malkauns, Bageshree, Bhoopali, Kalawati, Shree, Bheempalasi. Jog is
both notated (round 3) and a test2 raag; its notated recordings are kept out of test2's pool.

Whether the reader's fitted parameters generalise to raags it never saw is itself measured
(S7: the reader helps on notated raags and less elsewhere; tuned matcher costs transfer better).
