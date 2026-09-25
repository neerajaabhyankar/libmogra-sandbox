# pakad-matcher

## Problem Statement

Raags often have "pakad"s or phrases that belong to the mukhyanga -- that characterize it. A list of these phrases will appear in the `mukhyanga` section of the LibMogra raag database. We have audios for 50-ish raags in the Hugging Face dataset where we would expect to find these phrases. However there's no annotations or anything yet. I'd like to identify locations of a given phrase in a given audio clip, if it exists.

## Problem formalization

- Given an audio clip (known to be from some raag, for now) --> inferred Essentia pitch track (let's try to do this given only the pitch track)
- Given a phrase from its mukhyanga (e.g. "m, D, n, D" if the raag is Bageshree)
- Find time intervals where this is sung/played.

What makes this challenging:

1. The presence of kan swars and ornamentations (e.g., "m, D, (S`) n, D" should count)
2. Arbitrary time dilations (e.g., "m, D, n..... D" should count)
3. The above two often being part of the recommended ways a certain phrase is rendered -- i.e. just a string of notes doesn't capture what melody that phrase is supposed to stand for.
4. Presence in slow alaps as well as fast taans

## The larger goal (review, 2026-09-22)

Phrase-finding is a **contained proof-of-concept**. What this is really for: statistical questions
about a new recording, answered from its pitch track --

> "does ga mostly occur in the descent, or in the ascent too?" · "does this singer use a Re that
> isn't in the raag, and how often?" · "is this bandish sung with Kamod ang -- is that combination
> taken often?"

Two properties of that goal shape everything below: **rhythm is irrelevant**, and **being right 8
times in 10 is useful** -- an aggregate statistic tolerates per-note error, as long as the error is
not systematically biased.

---

## Where it stands

| | |
|---|---|
| Finding candidates | works -- ~2 in 3 of what it surfaces is accepted by ear, before ranking |
| Ranking them | precision@1 0.75, @3 0.86 after tuning -- but **tuned on what is now the test set**, so read it as an upper bound, not a result (see *Data discipline*) |
| Ranking, untuned | **precision@1 0.58, @3 0.58** -- the honest baseline, since those costs never saw a judgment |
| Reading a contour unaided | **misread rate 0.63** against the notation; it over-segments (see S6) |
| Test set, frozen | 168 y/n judgments, 12 samoohas, 6 raags (109 on recordings the training data does not touch) |
| Training data | notation corpus: 24 chunks, 111 stretches, **1023 swars**, 340 s |
| Corpus | 20.3 h of full recordings, pitch-tracked, tonic-annotated, train-split only |

Two lessons carry: **tuning the existing heuristic beat every new feature I invented** (S4 / S4b),
and **that tuning used the judgments that are now the test set**, so it has to be redone from the
notation corpus before it counts (S7).

---

## Data and representation

| fact | value | consequence |
|---|---|---|
| dataset v1.1 train | 1810 clips x 20 s ~ 10 h, 50 raags | the pinned, reproducible corpus |
| full recordings | 45 train-split videos of the 6 annotation raags, **20.3 h** | real context; read-only; test-split videos excluded throughout |
| **tonic is annotated** (`tonics.csv`) | best of 12 rotations was k=0 on 12/12 probed clips; frame-level in-scale 0.73-0.92 | the lever that dominated `motif-classifier` is *given* here, never re-estimated. The video id in each full-audio filename is the one `tonics.csv` annotates, so full recordings inherit it |
| mukhyanga phrases | 229 over 50 raags; 33 are 2-swar; 65 occur in >=10 raags | tiering is not optional (`phrases.py` keeps 159) |
| Melodia | ~60x real time; 10-cent quantised | 20 h tracked in ~12 min on 6 workers, in 5-minute blocks |

**The note segmentation had to be abandoned.** `melody-extraction/note_segmentation.py` was tuned
for the classifier (`min_note_dur=0.2`); its short-segment merge averages pitch *across* note
boundaries, so meend/kan transits land between swars:

| segmentation | in-scale (duration-weighted) |
|---|---|
| raw frames | 0.77-0.89 |
| default `min_dur=0.2` | 0.65-0.82 <- *worse than frames* |
| `min_dur=0.0` | 0.73-0.92 |

Verbatim `m D n D` occurs **0 times** in the collapsed note strings of 5 Bageshree clips. Symbol
matching is dead; everything works on the **frame-level tonic-relative cents contour**, where a
phrase is a path to align, not a string to find.

---

## The matcher

Subsequence Viterbi over a left-to-right chain (`matcher.py`): note *k* is `min_dwell_s` of chained
sub-states with a self-loop on the last (-> arbitrary dilation); between notes an ornament state
(-> kan swars, meend); free start and end. Octave-folded, with the phrase's own saptak marks used
for a register check. Candidates are re-scored with duration-invariant terms -- worst note's pitch
misfit + ornament-excursion fraction + gap fraction + wrong-step and register penalties -- so a slow
alap and a fast taan compete on equal terms. ~18 ms per 20 s clip; 27 min of audio in ~1 s.

**Every change below came from looking at a plot or a label, not from a metric moving.**

| # | change | found by |
|---|---|---|
| v2 | a span is scored by its **worst** note, not the mean | a Lalit "match" missing 2 of 7 notes scored like a real one |
| v3 | glides/kan within `kan_cents` of the neighbouring notes are **transit, not ornament** | fast Malkauns runs spent half their frames gliding and were charged for it |
| v4 | the **DP's** ornament emission uses the same band | the DP preferred cramming a note into 4 off-pitch frames over paying for a long glide, so real renderings never reached the candidate pool |
| v5 | **held notes** in an ornament slot are charged; **wrong steps** (direction/octave) cost +1 | Malkauns `n S (held g) m` matched Bageshree `,n S m`; a meend *down* to m matched `S`->m *up* |
| v6 | held-run detection credits the slope window | Melodia's 10-cent steps made a 170 ms held P look like motion |
| v7 | **tritones** accept either direction | `r -> P` is exactly 600c, where "shortest step" is ambiguous; ascending `r P` was being penalised |
| -- | candidate pool **scales with track length** (`CANDIDATES_PER_MIN`) | 20 regions is fine for a 20 s clip, hopeless for 30 min: Darbari `n m P `S` looked absent (3.4) until fixed, then scored 0.12 |
| -- | **register check** from the phrase's saptak marks | a `,n S m` an octave up scored 0.00 |
| S4b | five costs **tuned on the annotations** | see below |

---

## What was measured

### ✅ S1 -- the heuristic, eyeballed (`run_s1.py`, `results/s1/`)

Top candidates are, by eye, the phrase -- including andolit Darbari `m P d P d n P` and a 4-second
`G M d N d M m`. Median-band candidates are visibly forced. Two findings outlived the stage:
**the cost scale is not comparable across phrases** (genuine andolit Darbari 0.26-0.59; a flat
`S ,N r` 0.00), and **long phrases rarely appear whole** in 20 s chunks.

### ✅ S2 -- the label-free gate failed, and was the wrong gate (`run_s2.py`, `results/s2/v6/`)

Scoring each phrase against raags where it is merely *playable*: median AUC **0.55**, 8/159 phrases
passed the pre-registered gate. But those other raags' best matches are, by eye, **genuine
renderings** (`m D n D` in Alhaiya Bilawal and Jaijaivanti; `M d P` in Multani, Shree, Todi). Short
mukhyanga cells are shared melodic material, and the DB's document frequency counts only where a
phrase is *listed*, not where it is sung. So that AUC measures **exclusivity in performance, not
accuracy**, and no label-free null can separate the two. Scale-level separation does work (own 1.94
< playable 2.54 < unplayable 2.95). The run earned its keep by exposing three matcher bugs.

### ✅ S3 -- annotation, 168 labels (`annotate_app.py`, `annotations/`)

Settled by review: **(a)** judging a raag's character from a phrase is *not* the task -- Alhaiya
Bilawal having `m D n D` is irrelevant; **(b)** the raag label buys us a **searching ground** where
the phrase is likely. So: own raag only, and one question per candidate --

| verdict | meaning |
|---|---|
| **yes** | an *ornamented* path that still traces the phrase -- Neeraja would notate it as that phrase |
| **no** | an approximate presence she cannot identify as it: too ornamented (`m D n SRnS n D`), or a different phrase (`P D n D`) |

Phrases live in **`neeraja_mukhyangas.json`** -- hand-picked, sometimes shortened or modified,
overlapping the tanarang DB but not bound by it ("the DB phrases are a suggestion, not an airtight
signature"). 12 phrases, 6 raags: 7 verbatim + 5 Neeraja's own.

The pool: the matcher's best candidates from full recordings, <=3 per recording, spread over tempo
terciles so slow alap renderings are offered alongside fast ones (**no duration cap** -- that was an
arbitrary rule of mine that would have excluded exactly the slow renderings we want). Context is cut
at the surrounding silences -- the musical "sentence", 1.5-5 s either side.

The app: pitch track of the sentence, candidate shaded with its aligned path and swar labels, a
**playhead** locked to the audio, `z` zoom, `p` play just the candidate, `y`/`n`/`u`, and a free-text
note per judgment. Two bugs fixed mid-use: audio served without HTTP byte ranges (so the browser
could not seek at all), and one `timeupdate` listener leaked per press.

An earlier pool of 20 s clips (`annotations/labels_pool1.jsonl`, 24 labels) was scrapped -- snippets
of 1-2 s with no context. Not wasted: `,n S m` getting 12 no's out of 12 is what exposed the
octave-blindness and the fast-transit bias.

### ✅ S4 -- features built from the comments: negative (`features.py`, `s4.py`)

The comments sort cleanly, so I built one feature per reason: **salience** (Melodia confidence,
re-extracted over all 20 h) for "the S is coming from the drone" / "P from tanpura not voice"
(~14 of 59 no's); **tempo_ratio** against the local median note length for "too fast to be
meaningful, given the context"; **held_extra** for "this is n S g m".

| model | per-phrase AUC | P@3 |
|---|---|---|
| cost, as shipped then | 0.503 | 0.58 |
| logistic regression, unseen recording | 0.529 | 0.67 |
| logistic regression, unseen phrase | 0.548 | 0.64 |

- **Salience answers the wrong question.** Not degenerate (0-0.09, median 0.019 voiced) -- it
  measures how *strong* a pitch is, and a tanpura is strong and periodic. `salience_rel` = 1.04 on
  drone-flagged candidates, 1.04 on accepted ones.
- **`hpss+drone` separation is too destructive to verify with.** Per-candidate windows, re-tracked:
  disputed notes do vanish (the tanpura P in `r P r G r S`), but so do genuine ones -- notes lost or
  off by >60c went 5 -> 19 on flagged candidates and **1 -> 11 on accepted ones**. The version worth
  trying is a separator that knows Indian instruments (BS-RoFormer on Saraga; see
  `source-separation/plan.md`), not HPSS.
- **In-span features cannot settle the "different phrase" cases.** `held_extra` fires on 1 of the 5.
  Re-reading them, `,n S m` is called `n S g m` because of *what follows* the match, and "the m here
  is actually a kan in n S (m) g" was still marked yes on technicality. That judgment lives in the
  surrounding movement, which nothing confined to the span can see.

### ✅ S4b -- tuning the existing heuristic: positive (`tune.py`)

Coordinate ascent over the re-scoring knobs on the fixed labelled spans, objective = per-phrase AUC.

> **Superseded as a headline by the 2026-09-24 data discipline.** These judgments are now the test
> set, so a number fitted on them cannot be reported as performance. The section stays because
> *what* tuning changed is still the finding -- especially `note_trim` -- and because the size of
> the gain says how much a fitted model should be expected to buy.

| | per-phrase AUC | P@1 | P@3 |
|---|---|---|---|
| as shipped | 0.500 | 0.58 | 0.58 |
| **tuned, leave-one-phrase-out** | **0.669** | **0.75** | **0.86** |
| tuned on all 12 (optimistic) | 0.719 | | |
| adopted defaults (register kept) | 0.682 | 0.75 | 0.78 |

What each change is worth alone: `note_trim` 0.5 -> **1.0** (0.646), `register_penalty` 0.5 -> 0
(0.603), `free_cents` 30 -> **15** (0.563), `held_slope` 400 -> **800** (0.518), `note_cap` 3 -> **2**.

The interesting one is `note_trim`. Scoring a note on its **best half** of frames -- my andolan
tolerance -- let a note that is merely *passed through* count as sung. The ear wants the note
actually dwelt on. `register_penalty` -> 0 is **not** adopted: the pool it tuned on was built with
the register check on, so tuning never saw the octave-wrong candidates it removes.

Per-phrase yes-rates: `M P d M G r` 0.86 · `,n S g m P`, `,n S g R S`, `m D n D` 0.79 · `,n S m`,
`,n D ,N S`, `m P d n P` 0.71 · `g m R S`, `M G M r G` 0.64 · `r P r G r S` 0.57 · `,N r G M P` 0.50
· `n m P `S` 0.07. The last is **kept**: Neeraja recognises the phrase generally, it just isn't in
these recordings, and the negatives are themselves signal.

---

---

## The task, stated so it can be scored

**Input** a pitch track (or audio plus its tonic in Hz) · a **samooha**: 2-8 swars, optional saptak
marks, e.g. `,n S m`. **Output** ranked time intervals, each with a cost and a calibrated
probability. **Out of scope, deliberately**: rhythm, raag identification, and whether the samooha is
characteristic of anything.

| | |
|---|---|
| unit of evaluation | an interval; a hit needs IoU >= 0.5 with a human-marked occurrence (boundaries are genuinely fuzzy) |
| primary metric | **precision@1 and @3** -- what a user of the tool feels |
| ranking metric | **per-phrase AUC** -- does a yes outrank a no *within* one samooha |
| threshold metric | precision/recall at one **global** threshold, for "find all occurrences" |
| recall | only measurable against notated chunks (S5b); everything to date is precision-only |
| held-out discipline | leave-one-phrase-out (an unseen samooha) and grouped by recording |

**Where it stands against that.** P@1 0.75 · P@3 0.86 (leave-one-phrase-out) · per-phrase AUC 0.669
· at the best global threshold, F1 0.82 (precision 0.71, "recall" 0.96 -- over proposed spans only,
so it is an upper bound, not recall). **Next targets**: P@3 >= 0.90, and once notation exists,
recall >= 0.80 at precision >= 0.80.

**Testbed**: the 12 samoohas in `neeraja_mukhyangas.json` plus whatever the notated chunks yield.
Statistical queries (the long-term goal) stay out of the evaluation until there is a corpus to score
them on -- the phrase task is the proxy that can be scored today.

## The tool (`pakad.py`)

The formulation is meant to be handed to a bigger system, so it has one small surface:

```python
from pakad import find
find("raga.mp3", ",n S m", tonic_hz=155.06, top_k=5, min_probability=0.6)
# -> [<,n S m 464.18-464.44s p=0.67>, ...]
```

`poetry run python pakad.py raga.mp3 --samooha ",n S m" --tonic 155.06 --top 5 [--json]` does the
same from a shell. It takes audio, a `Contour`, or a raw `(f0, hop)` pair, so a caller that already
has a pitch track never re-tracks. **The tonic is required and never guessed** -- it is the one
input that changes every answer.

`probability` comes from `calibrate.py`: Platt scaling of the cost on the 168 judgments, validated
leave-one-phrase-out (Brier **0.208** against a 0.228 base rate). It is honest but coarse -- 
reliability by band is 0.00 / 0.62 / 0.47 / 0.69 / 0.80 -- so treat it as "roughly how sure", not a
probability to do arithmetic with. It will sharpen when the corpus grows.

---

## Data discipline

**See [`DATA.md`](DATA.md)** for the glossary (judgment, notation, chunk, stretch, candidate, pool,
misread rate, ...) and the full rules, and run **`poetry run python audit.py`** for the live state:
it computes the splits from the files and fails if a rule is broken.

The short version, because it changes how every earlier number reads:

| | |
|---|---|
| **training** | **notations** -- what a musician heard in a stretch, written as swars. 24 chunks, 111 stretches, **1023 swars** |
| **test** | **judgments** -- one y/n per candidate span, on recordings the notation never touches: **109** over 12 samoohas |
| **validation** | judgments on notated recordings but at *different moments*, 5 s margin: **58**. Neeraja's suggestion, and it rescues a third of the judgments from being wasted |
| **set aside** | 1 judgment that overlaps a notated stretch |
| **awaiting judgment** | **9 samoohas, 122 candidates** over Des, Tilak Kamod, Multani, Todi, Bhinna Shadja |
| **awaiting notation** | **24 chunks, 420 s** over Yaman, Bhairav, Malkauns, Bhoopali, Jog, Kalawati -- fresh raags, two of them audav |

**Today's headline phrase numbers were tuned on what is now the test set.** S4b fitted five costs by
coordinate ascent on those judgments and `calibrate.py` fitted the probability on them, so
P@1 0.75 / P@3 0.86 / per-samooha AUC 0.669 are optimistic by an unknown amount and do not survive
the discipline. The baseline to beat is the **untuned** matcher's P@1 0.58 / P@3 0.58, whose costs
never saw a judgment. Under the new rules every parameter comes from notation, validation tunes,
and the test is scored once.

**What the notation corpus can already estimate** (882 notated notes with an alignment):

| quantity | corpus says | what it sets |
|---|---|---|
| intonation, \|median pitch - equal-tempered target\| | median **23 c**, 75th 43 c, 90th 96 c | `free_cents`, `scale_cents` -- the phrase matcher's tuned 15 c is *tighter than the median note*, a sign it was fitted to separate candidates rather than to describe singing |
| note duration | median **0.09 s**, 10th pct 0.05 s | `min_dwell_s`, and a duration model |
| transit/ornament share of a stretch | median **0.15** | `orn_cost`, `kan_cents` |
| per-swar deviation from equal temperament | S -1, P +2, R +5, G +5, M +3 · **d +25, g +18, N +17, D +15, n +12** | per-swar emissions -- and this is the shruti question itself, answered from Neeraja's own ear |

The komal swars sitting sharp of equal temperament, while S and P sit on it with the tightest
spread (IQR 20-41 c against 33-58 c), is the first musical result this corpus has produced.

## What's next

The goal above changes the target: the core capability is **reading a contour as a swar sequence**,
well enough that *aggregates* over it are right. Phrase-finding is one query against that reading.

### ✅ S5a -- likelihood ratio: negative (`decode.py`, `s5a.py`)

The score is **absolute** -- it says how well a span fits a phrase, not whether the phrase is the
best account of that span. A stretch sitting quietly on two swars fits half the database at ~0.
So: score a span by `phrase-constrained decode - free decode` of the same span, under identical
emissions, ornament rules and dwell. Both decodes now exist in `decode.py` and cover the span end
to end.

| score | per-phrase AUC | P@1 | P@3 | pooled AUC (one global threshold) |
|---|---|---|---|---|
| tuned cost | **0.682** | 0.75 | 0.78 | **0.692** |
| ratio, free = any of 12 swars | 0.644 | 0.58 | 0.69 | 0.664 |
| ratio, free = the raag's scale | 0.647 | 0.58 | 0.69 | -- |
| ratio, free = scale, per note | 0.654 | 0.58 | 0.67 | -- |
| cost + ratio | 0.689-0.702 | 0.75 | 0.75-0.81 | 0.702 |

It does make the score slightly more comparable across phrases (spread of the per-phrase median
of accepted candidates: 0.73 vs 0.96) but not enough to matter: at the best global threshold both
reach F1 0.82. **Not adopted as the score.**

**The caveat that keeps the idea alive.** Every labelled span was *chosen by the matcher* as one of
its best fits, so the constrained and free decodes almost agree there by construction. What the
ratio is actually for -- deciding, over a whole recording, which stretches are the phrase and which
are nothing -- is untested, because we have no labels for spans the matcher never proposed. That is
exactly the gap the notation corpus fills. `decode.py` stays: the free decode is also what the
notation view aligns with, and it is the skeleton of the learned model in S7.

### ✅ S5b -- the notation corpus: 24 chunks notated

Notate 15-30 s chunks **as heard**: swar sequence, octave marks, no rhythm. Why this beats more
y/n labels: a y/n is **one bit**, a notated 20 s chunk is 30-60 notes; it gives **recall**, which
candidate-verification structurally cannot; it is ground truth for the reading itself; and phrase
positives and negatives fall out of it for *any* samooha, so "more phrases / more raags" stops
being a separate annotation job.

**Chunks** (`chunks.py`): 24 stretches, 2 recordings per raag, one slow and one dense from each --
0.10-0.55 held notes/s for the alap chunks, 1.25-2.35 for the taans (density measured with the same
`_held` the scorer uses, so "slow" and "dense" mean what the model sees). ~7 minutes of audio.

**The view** (`/notate`), after the 2026-09-23 review:

| | |
|---|---|
| **sub-ranges, not whole chunks** | drag across the plot to select a stretch; a taan gets split into 4-5 of them "for convenience + clarity + correction where your alignment is wrong". Stretches are listed under the plot; click one to reopen it for editing (swars *and* edges), `esc` to leave it, `×` to drop it |
| **read, don't search** | notation spans the selection (rim absorbed); the matcher's tight candidates are a fallback only, since as options they crowd every swar into one sweep that passes through all of them |
| **ties split evenly** | `R R` over a stretch that just sits on R costs the same however the boundary falls, so Viterbi gave one note everything and the other a few frames. Same-swar runs with nothing between are split evenly; boundaries the contour actually marks, and genuinely unequal durations, are left alone |
| **notation's own costs** | the matcher's tuned `free_cents=15, note_trim=1.0` demand near-perfect intonation on every frame -- right for phrase matching, wrong here, where a dip that leans on a swar *is* that swar. `NOTATE_MATCH` relaxes to `free_cents=35, scale_cents=70, note_trim=0.4, min_dwell=0.05`, on-held reward down to 0.2 |
| **`space evenly`** | when the tracker did not follow the instrument at all, type what you hear and space it evenly; stored as `method: "even"`, shown as *spaced by hand*. Evidence of **what was sung, not when** -- never to be used for scoring timing |
| **coverage counts, but notes land on notes** | free ends inside the selection (silence, drone), scored `cost + NOTATE_COVER_WEIGHT x (1 - coverage) + NOTATE_HELD_WEIGHT x (1 - on-held)`; the span-covering candidate runs with **absorbing rim states**, so an unaccounted-for blip at an edge costs like ornament instead of dragging a note out to it. Coverage is always shown. On the worked example, `g g g m m` places all three *g*s on held pitch (300/315/308 c) over 90 % of the selection |
| **one swar is a legal notation** | useful exactly when the alignment fails and you want to pin a single note down |
| **a swar keypad** | `,P` to `` `P ``, laid out like a keyboard with every natural a step apart and komal/teevra between their neighbours; saptaks shown by a band behind madhya, not by gaps. Clicking appends. Raag notes can be **marked by hand** for a visual ring -- set by the notator, never inferred |
| dropped | the "has non-voice pitch" flag -- drone and stray pitch are always there, so the flag carried nothing |

Alignment reuses the matcher itself (`matcher.match` over the selection), so the notator sees
exactly what the model would propose, and correcting it is the supervision.

Interaction settled on review: colour says state (aligning blue, added green, ornament purple --
a category, not a fault); `⇧space` / `⇧K` drive playback mid-word so it never fights typing; a
selection plays **once** and stops at its end, and its edges can be dragged to extend it;
`add stretch` sits beside `align` so the pending action is visible; no flags.

**The corpus**: 24 chunks notated, **102 stretches, 845 swars, 304 s** of notated audio (of 420 s
offered); 93 stretches aligned to the track, 9 spaced by hand. Three chunks came back empty and
were replaced from other recordings (`chunks.py --replace`), awaiting notation.

**Still open**: per-note correction (deferred -- sub-ranges may make it unnecessary), and whether
20 s / 15 s chunks are the right size. Further ideas for the tool live in
**`notator.md`**, which is its own parking lot now that it is worth more than this errand.

### ✅ S6 -- scoring the reading against the notation: over-segmentation is the wall (`s6.py`)

`decode.free_read` decodes a stretch with **no phrase to guide it** -- the automatic reading --
and the corpus scores it. The metric is the **misread rate**:
`(substitutions + deletions + insertions) / notated swars` -- the same shape as word error rate in
speech. 0 is perfect; 1 means as many mistakes as there are notes. Insertions and deletions are
always reported separately, because they fail in opposite directions and an aggregate hides which.

**The first run said the model was not close.**

| | notated | read | sub | del | ins | misread rate |
|---|---|---|---|---|---|---|
| everything | 845 | **1780** | 248 | 31 | **966** | 1.47 |
| alap | 146 | 586 | 23 | 0 | 440 | 3.17 |
| taan | 699 | 1194 | 225 | 31 | 526 | 1.12 |

The machine hears nearly everything Neeraja does (31 deletions) and then **more than twice as much
again**. It reads every pitch region a glide passes through; she writes the notes that were
*intended*. Alap is four times over-read, because a slow meend wanders through many swars that
nobody notates.

**The missing lever**: declaring a new note cost nothing. Adding `onset_cost` to the free decode
and fitting it on the corpus (`s6.py --sweep`) -- the first parameter this project learned from
data rather than hand-set -- gives:

| | notated | read | sub | del | ins | misread rate |
|---|---|---|---|---|---|---|
| everything | 845 | 586 | 146 | 322 | 63 | **0.63** |
| alap | 146 | 172 | 38 | 15 | 41 | 0.64 |
| taan | 699 | 414 | 108 | 307 | 22 | 0.63 |

**One knob only trades one error for the other.** misread rate is flat at 0.63-0.64 across a wide range of
(`onset_cost`, `min_dwell`); pushing insertions from 966 to 63 costs 291 deletions. That flatness
is the finding: the note-segmentation *model*, not its parameters, is the limit. Fitting the onset
per tempo helps a little and confirms the tempo dependence -- alap wants 3.0 (misread rate 0.53), taan wants
1.5 (misread rate 0.60) -- which points at making it a function of local note density rather than a constant.

**Which statistical questions are answerable today**, measured the way S6 was supposed to be:

| question | notation vs reading | verdict |
|---|---|---|
| which swars are used in this chunk | recall **0.95**, precision **0.72** (6 missed, 46 spurious over 21 chunks) | usable with care; the spurious ones are transit swars |
| how *often* each swar is used | swar-histogram total variation, median **0.24** per chunk; 11/21 chunks under 0.25 | not yet -- the over-read is not unbiased |
| is this swar approached from below or above | P .49/.51, n .44/.44, G .50/.48, g .26/.34 agree; m .35/.58, r .35/.61 do not | per swar, and only for the ones that are dwelt on |

The 8-in-10 tolerance is the right frame and it is **not met yet** for counting questions. The
bias is systematic, not noise: transit notes inflate exactly the swars that sit between other
swars, which is why `m` and `r` are the two that disagree most.

### 🟨 S7 -- fit the reader on the notation corpus, then score the frozen test once

Everything hand-set becomes estimated, from training data only:

1. **Emissions** `P(cents | swar)`: per-swar centre and spread, straight from the 882 notated notes
   (the table above). Replaces `free_cents` / `scale_cents` and gives andolan-heavy swars their own
   width rather than one global tolerance.
2. **Durations**: a per-swar duration distribution replaces `min_dwell_s`, and gives the decode a
   reason to prefer a plausible note length over a 3-frame sliver.
3. **Onset cost as a function of tempo**: fitted per tempo it already wants 3.0 for alap and 1.5 for
   taan; local note density is measurable without labels, so this becomes a function, not a constant.
4. **Ornament/transit occupancy** from the measured 0.15 share, replacing `orn_cost` and `kan_cents`.

**Evaluated in two places, neither of which is the frozen test:**
- **reading quality**, cross-validated *within* the notation corpus, leaving out whole recordings --
  misread rate, insertions and deletions separately, alap and taan separately;
- **statistic agreement** (the S6 table), same folds.

Only when that is settled does the phrase test get scored, once: P@1 / P@3 / per-phrase AUC on the
109 disjoint judgments, against the untuned baseline of 0.58 / 0.58.

**Bootstrapping, once the reader is fitted**: run it over the unnotated train recordings, keep only
stretches it reads confidently, and refit on notation + those pseudo-notations. The check that it is
not just amplifying its own bias is the same held-out misread rate -- if pseudo-labels help, it
improves; if it is feeding on itself, it will not.

A neural sequence labeller stays out until this is done. 1023 swars can fit per-swar emissions and a
duration model honestly; it cannot train an encoder.

### What I need from Neeraja

1. **Confirm the split** — judgments = frozen test, notations = train; headline scored on the 109
   judgments whose recordings the notation corpus does not touch.
2. ~~More samoohas for the test set~~ **done 2026-09-24**: 8 added over Des, Tilak Kamod, Multani
   and Todi; 8.5 h of those raags pitch-tracked; **108 candidates built and waiting** in the app.
3. **More notated chunks for training**, from **fresh raags** -- Yaman, Bhairav, Malkauns, Bhoopali
   plus the audav pair Jog and Kalawati (`config.NOTATION_RAAGS_R3`). The six original raags have
   almost no unjudged recordings left, and new raags widen the per-swar coverage that the emission
   fits need.
4. Nothing else. S7 fits on what exists; more data makes the fits sturdier, it does not unblock them.

### Annotation, in priority order

1. **More phrase judgments, on reserved test recordings** -- the test set is 12 samoohas and 168
   calls, and it is now the only thing standing between us and a self-graded model. Neeraja has
   offered more phrases and raags; each new samooha is ~14 candidates.
2. **More notated chunks** (train), from recordings *not* reserved for test.
3. Judgments on recordings the notation corpus uses are worth less -- they can only be a secondary
   number.

---

## Files

| file | what |
|---|---|
| `config.py` | every constant, including which ones were tuned and on what |
| `contour.py` | f0 cache for the pinned clips; `contour()` -> tonic-relative cents at ~56 fps |
| `fullaudio.py` | the full recordings: index by video id, inherit the annotated tonic, f0 + salience cache |
| `phrases.py` / `mukhyangas.py` | the DB catalogue with tiering / Neeraja's hand-picked phrases |
| `matcher.py` | the model: `match()` and `score_path()` |
| `pool.py` | annotation pools from full audio, with sentence-length context |
| `annotate_app.py` + `.html` | the local annotation app (playhead, zoom, comments) |
| `s3.py` | the earlier terminal annotation loop (pool v1) |
| `features.py` / `s4.py` | features from the comments, and their evaluation |
| `decode.py` | phrase-constrained and free decodes of a span; `align()` for notation |
| `calibrate.py` | cost -> probability (Platt, leave-one-phrase-out) |
| `pakad.py` | **the tool**: `find(audio, samooha, tonic_hz)`, plus a CLI |
| `chunks.py` / `notate_app.html` | notation chunks and the notation view (see `notator.md`) |
| `s5a.py` | the likelihood-ratio evaluation |
| `tune.py` | coordinate ascent over the costs on the labels |
| `audit.py` | **the data discipline, executable**: computes the splits, checks the rules |
| `DATA.md` | the glossary and the rules in prose |
| `run_s1.py` / `run_s2.py` / `plot.py` | the eyeball run, the null-control run, plotting |

Reused from `../raag-identifier/`: `utils.config`, `utils.dataset`, `utils.raagdb`,
`utils.extract._essentia` (settings; `fullaudio._melodia` re-implements it to keep salience),
`source-separation` (tested, not adopted). **Not** used: `melody-extraction/note_segmentation.py`,
deliberately. Nothing outside `../raag-identifier/` is imported.

---

## Log

- **2026-09-20** -- Probed the data: annotated tonics land notes on the right swar grid (12/12);
  the shared note segmenter's merge *lowers* duration-weighted in-scale (0.67 vs 0.90); no verbatim
  `m D n D` survives in 5 Bageshree clips. All three push the project onto the frame-level contour.
  S0 (f0 cache, 159/229 phrases kept) and S1 (matcher + plots over 8 raags) done; four scoring
  revisions, all driven by plots.
- **2026-09-20** -- S2 ran: 8/159 phrases pass the pre-set gate. Diagnosis: the "playable raag" null
  is full of genuine occurrences, so the gate measured exclusivity, not accuracy. Three matcher bugs
  found and fixed along the way.
- **2026-09-22** -- S3: `neeraja_mukhyangas.json`, terminal loop, pool v1 (24 labels) scrapped after
  review; pool v2 built from 20.3 h of full recordings with a visual annotation app. Bugs found:
  octave-blind matching, tempo-skewed candidates, candidate pool far too small for long recordings,
  tritone steps penalised, audio served without byte ranges.
- **2026-09-24** -- Data discipline set: the phrase judgments become the **frozen test set**, the
  notation corpus is the **training data**. That retires S4b's tuned numbers as headlines (they were
  fitted on those judgments) and makes the untuned 0.58 / 0.58 the baseline to beat. 35 % of the
  judgments share a recording with the notation corpus, so the headline test shrinks to the 109
  disjoint ones. Probed what the corpus can estimate: intonation spread (median 23 c), note duration
  (median 0.09 s), transit share (0.15), and **per-swar deviation from equal temperament** -- komal
  d +25 c, g +18 c against S -1 c, P +2 c, which is the shruti question answered from the notation.
  Three replacement chunks notated; corpus now 1023 swars.
- **2026-09-24** -- **notation corpus done**: 24 chunks, 102 stretches, 845 swars, 304 s. S6 run:
  the free reading over-segments badly (1780 notes read against 845 notated, 1.47). Added an
  `onset_cost` to the decode and fitted it on the corpus -- the project's first learned parameter --
  reaching misread rate 0.63, but the knob only trades insertions for deletions, so the segmentation model
  is the limit. Aggregates: "which swars" recall 0.95 / precision 0.72, histogram TV 0.24 median,
  ascent-descent agrees for dwelt-on swars only. Three empty chunks replaced from other recordings.
- **2026-09-23** -- S5a likelihood ratio: **negative** (0.644-0.702 vs 0.682 for the tuned cost),
  with the caveat that the labelled spans are the matcher's own picks, so the comparative question
  it was built for is untested until recall data exists. Task restated so it can be scored
  (P@1/P@3, per-phrase AUC, one global threshold; recall pending notation). Tool shipped:
  `pakad.py` + calibrated probability. Notation chunks and the `/notate` view built, then reworked
  on review: sub-range selection, free-ended alignment with coverage reported, and a `,P`-to-`` `P ``
  swar keypad. `notator.md` opened for where the notation tool goes next.
- **2026-09-22** -- **168 labels done.** S4: every feature invented from the comments lands at
  0.53-0.55 per-phrase AUC; salience cannot tell drone from voice, and HPSS separation destroys
  genuine notes as fast as spurious ones (both measured). S4b: **tuning the five existing costs
  reaches 0.669 / P@3 0.86**, adopted. Goal clarified -- statistical queries over a pitch track, with
  phrase-finding as the proof-of-concept -- and the roadmap rewritten around a notation corpus.
