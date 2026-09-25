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

Terms are grouped by what they belong to, not alphabetically, because they only make sense in
relation to each other.

### The audio

| term | meaning |
|---|---|
| **recording** | one full performance, e.g. `8ldBWSCfR0Q`. Named by its YouTube id, which is also how `tonics.csv` annotates it. Everything is split by recording, never inside one |
| **clip** | a 20-second excerpt in the pinned HuggingFace dataset. Used for the earlier stages; the annotation work uses full recordings instead |
| **tonic** (Sa) | the reference pitch of a recording, in Hz, annotated by hand in `tonics.csv`. Never estimated here |
| **contour** | the pitch track: one number per 18 ms frame, in **cents above Sa** (100 cents = one semitone), or "unvoiced" where no pitch was found |
| **saptak** | octave. `,n` is mandra (below Sa), `n` is madhya (the middle), `` `n `` is taar (above) |

### What a musician contributes

| term | meaning |
|---|---|
| **samooha** (or phrase, pakad) | a short sequence of swars, e.g. `m D n D`. Listed in `neeraja_mukhyangas.json` |
| **candidate** | a span of audio the matcher proposes as an occurrence of one samooha: a recording plus a start and end time |
| **judgment** | **one y/n on one candidate**: "is this really that samooha?". Made in the phrase app (`/`). Stored in `annotations/labels.jsonl`. **These are test and validation data** |
| **pool** | the fixed set of candidates offered for one samooha (14 of them). A judgment is stored as *an index into its pool*, so **a pool must never be rebuilt once judged** |
| **chunk** | a 15–20 s stretch of a recording chosen for notating, listed in `annotations/chunks.json` |
| **stretch** | a sub-range of a chunk that has been notated: times, the swars heard, and how they were placed |
| **notation** | the swars a musician heard in a stretch, written out. Stored in `annotations/notations.jsonl`. **This is training data** |
| **aligned / spaced by hand** | how a stretch's swars were placed in time: `align` fits them to the pitch track, `even` spreads them evenly because the tracker did not follow the instrument. A hand-spaced stretch is evidence of *what* was sung, not *when* |

### What the machine produces

| term | meaning |
|---|---|
| **reading** | what the model hears in a stretch with no samooha to guide it: a swar sequence, from `decode.free_read` |
| **alignment** | the model's placement of a *given* swar sequence onto a contour — used by the notation app, and by the matcher when it scores a candidate |
| **cost** | the matcher's score for a candidate. Lower is better. Not comparable across samoohas |
| **fit** | the same number, shown in the notation app for a stretch |
| **coverage** | how much of a selected stretch the alignment actually accounts for |
| **misread rate** | `(substitutions + deletions + insertions) / notated swars`, the same shape as word error rate in speech. 0 is perfect; 1 means as many mistakes as notes. Insertions and deletions are always reported separately, because they fail in opposite directions |
| **P@1, P@3** | of the 1 or 3 candidates the matcher ranks highest for a samooha, how many a musician accepted. Averaged over samoohas |
| **per-samooha AUC** | does the score rank a "yes" above a "no" *within one samooha*. 0.5 is chance |

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
| **R1** | a judgment whose recording carries no notation is **test** |
| **R2** | a judgment on a notated recording, at a different moment (5 s of margin), is **validation** — notation teaches the model about a moment, not about a whole recording |
| **R3** | a judgment overlapping a notated stretch is **unusable**: neither fitted on nor scored |
| **R4** | raags in `config.TEST_ONLY_RAAGS` are never notated, so all their judgments are test |
| **R5** | a pool is never rebuilt once it carries judgments — labels are indices into it, so regenerating one silently re-points them. `pool.py` refuses without `--force` |

R2 is what makes the annotation effort pay: without it, every judgment sharing a recording with
notation would be wasted. As of 2026-09-24 that is 58 judgments rescued as validation, 1 set aside.

**Guards in code, not just in prose:**

- `pool.py` refuses to rebuild an existing pool (R5).
- `chunks.py` skips `TEST_ONLY_RAAGS` and any recording that already carries a judgment (R4, and
  keeps R1 growing rather than shrinking). It also **adds** chunks rather than rewriting
  `chunks.json`: notations refer to chunks by id, and a chunk's `video`/`t0` is what turns a
  stretch's times into recording times, so rewriting one orphans the work done on it.
- `audit.py` recomputes R1–R5 from the files and fails if any is broken.

**Consequence worth remembering:** 42 recordings now carry judgments, which leaves almost nothing
free in the six originally notated raags. Further notation therefore comes from **fresh raags**
(round 3: Yaman, Bhairav, Malkauns, Bhoopali, Jog, Kalawati). The reader's parameters are
raag-independent, so this is not a compromise — it tests whether they generalise.

---

## Where it stands (regenerate with `audit.py`)

| | |
|---|---|
| training, notation | 24 chunks · 111 stretches · **1023 swars** · 340 s · 6 raags · 15 recordings |
| test, judgments | **109** over 12 samoohas, 27 recordings |
| validation, judgments | **58** over 12 samoohas, 15 recordings |
| set aside (R3) | 1 |
| awaiting judgment | **9 samoohas, 122 candidates** — Des, Tilak Kamod, Multani, Todi, Bhinna Shadja |
| awaiting notation | **24 chunks, 420 s** over Yaman, Bhairav, Malkauns, Bhoopali, Jog, Kalawati (round 3) |

Anything fitted before 2026-09-24 — the tuned matcher costs in `config.MATCH` and the probability
in `results/calibration.json` — was fitted on what is now the test set, and is marked as such in
`plan.md`. Those numbers do not count until they are re-derived from training data.
