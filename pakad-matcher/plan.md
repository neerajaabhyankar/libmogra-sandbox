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

---

## What the data actually looks like

Measured 2026-09-20 with throwaway probes (Bageshree / Bhoopali / Malkauns, train split). These numbers set the design.

| fact | value | consequence |
|---|---|---|
| dataset v1.1 train | 1810 clips × 20 s = **~10 h**, 50 raags, 8 videos/raag, 40 clips/raag | enough to find phrases; too much to annotate exhaustively |
| **tonic is annotated per clip** (`tonics.csv`) | verified: best of 12 rotations was k=0 on **12/12** probed clips, frame-level in-scale 0.73–0.92 | the lever that dominated `motif-classifier` is *given* here. Do not re-estimate it. |
| mukhyanga phrases | **229** over the 50 raags, mean 4.6/raag | |
| phrase lengths | 2 swars: 33 · 3: 54 · 4: 59 · 5+: 83 | a third of "phrases" are 2-grams |
| phrase specificity (DB document frequency) | **90/229 unique to one raag**; **65/229 occur in ≥10 raags** (`DP`, `mP`, `NS`, `RS`…) | `Kafi: m P` is not a locatable event. Phrase tiering is not optional. |
| Essentia Melodia cost | ~60× real time on M1 | full train f0 cache ≈ **10 min**. Cheap. |

### The one result that changes the representation

`../raag-identifier/melody-extraction/note_segmentation.py` was tuned for the classifier
(`tol_cents=50, min_note_dur=0.2`). Its short-segment merge **averages pitch across note
boundaries**, so meend/kan transits land *between* swars:

| segmentation | notes/clip | in-scale (count) | in-scale (**duration-weighted**) |
|---|---|---|---|
| frames, no segmentation | — | — | **0.77 – 0.89** |
| `tol=50, min_dur=0.2` (the default) | 17–47 | 0.60–0.75 | 0.65–0.82 ⟵ *worse than frames* |
| `tol=50, min_dur=0.0` | 62–352 | 0.49–0.71 | **0.73–0.92** |
| `tol=30, min_dur=0.1` | 23–101 | 0.74–0.87 | 0.78–0.90 |

Two readings:
- the off-scale notes are the **short** ones (count-weighted ≪ duration-weighted) — they are
  ornament transit, not error;
- **the default merge destroys exactly the information we need.** Do not reuse the cached
  `notes` arrays. Cache the **frame-level f0** and work on the contour.

Corollary, measured: the verbatim string `m D n D` occurs **0 times** in the collapsed note
strings of 5 Bageshree clips. Symbol-string matching is dead on arrival — consistent with
`motif-classifier`'s M1 (0.20 of clips contained any verbatim mukhyanga) and its M5 premise.

### So the shape of the solution is forced

Work on the **tonic-relative cents contour**. A phrase is not a string to find; it is a
*path* to align. That is a subsequence alignment / left-to-right HMM decode — which gives
challenges 1, 2 and 4 for free:

| challenge | mechanism |
|---|---|
| 2. arbitrary time dilation | per-state self-loop, unbounded dwell |
| 1. kan swars, ornamentation | an "ornament" state between phrase states that absorbs any pitch at a fixed cost |
| 4. alap vs taan | score normalised by matched duration, not frame count |
| 3. melodic identity, not note string | ⟵ **this is the part that needs human data.** Stages 3–5. |

---

## Approach

Six stages. Each one is gated on the previous producing a number worth continuing from.
Stage 2 is the gate before we spend any of your time annotating.

### ✅ S0 — scaffolding and the f0 cache

| file | what |
|---|---|
| `config.py` | every constant (paths, tiers, matcher costs, plot style) |
| `_bootstrap.py` | puts `../raag-identifier` on `sys.path` |
| `contour.py` | `python contour.py` builds `cache/f0_essentia_v1.1_train.npz` (1810 clips, raw f0 only, ~4 min on 6 workers). `contour(clip_id)` → tonic-relative cents, downsampled 225 → 56 fps by NaN-aware median |
| `phrases.py` | `python phrases.py` → `results/phrases.csv`: all 229 phrases, `kept` flag, `df`, `idf`, `turns`. **159 kept** (≥3 swars, full-phrase DF ≤ 9); `idf` = mean IDF of 2-/3-grams, so `G D P` outranks `G m P` (Q4) |

### ✅ S1 — the heuristic matcher (no learning)

`matcher.py`, `plot.py`, `run_s1.py`. ~18 ms per (clip, phrase); all 28 focus phrases × their clips in ~1 min.

**Model.** Subsequence Viterbi over a left-to-right chain:
note *k* = `min_dwell_s` (70 ms) of chained sub-states, the last one with a self-loop; between
notes an ornament state; free start and end. Octave-folded throughout.

**Scoring, as it evolved** (each step came from looking at plots, not from a metric):

| version | change | why |
|---|---|---|
| v1 | re-score = mean per-note misfit + ornament fraction | DP total cost is length-biased; re-score is duration-invariant (alap vs taan) |
| v2 | per note: mean of its **best half** of frames; phrase: **worst note** | Lalit rank-2 was `M d M m` with G, N crammed into 4 off-pitch frames — scored like a real one. Worst-note = "every note must be present". Best-half = andolit Darbari *d* survives |
| v3 | glides that stay between the neighbouring notes ±`kan_cents` (200) are **transit, not ornament** | fast Malkauns runs spent ~50 % of frames gliding/overshooting (kan) and were charged for it — the `m D (S') n D` case the problem says should count |
| v4 | the **DP's** ornament emission uses the same band (`transit_cost` 0.1 vs `orn_cost` 0.6) | the DP still preferred cramming a note over paying for a long glide, so real renderings never reached the candidate pool. Clips with a candidate < 0.4: `,n S m` 16 → 25, `M d P` 16 → 20, `` `g `S n d `` 12 → 25 |

**What the plots show** (`results/s1/plots/<phrase>.png`: 6 best, ≤ 2 per video, then 2 from
the median for contrast; `results/s1/audio/`: top 3 as wav with 1.5 s context):

- Top-ranked candidates are, by eye, the phrase: `m D n D`, `m P d P d n P` (andolit *d* and
  all), `G M d N d M m`, `S ,N r`. Median-band candidates are visibly forced. **Ranking within
  a phrase works.**
- **The cost scale is not comparable across phrases.** Genuine andolit Darbari scores 0.26–0.59;
  a flat `S ,N r` scores 0.00. Any threshold has to be per phrase → S2's per-phrase null.
- **Long phrases (≥ 7 swars) rarely appear whole** in 20 s chunks: median best = 3.0 (a note
  entirely missing). Malkauns `g m n d m` found in 6/50 clips. Accepted (review: the DB is a
  suggestion, not a signature).
- **Short phrases light up everywhere**: `S ,N r` < 0.2 in 35/45 Shree clips. Some are gamaks
  around Sa that the ±200 c kan band waves through as "transit". Whether that is the phrase or
  just Sa-territory is exactly what S2 must answer.

Summary: `results/s1/summary.csv`; every candidate: `results/s1/candidates.csv`.

Known issues, deferred to S4 tuning: kan band is fixed at 200 c regardless of the step size;
octave-aware matching not implemented (flag reserved); plots fold octave errors silently.

### 🟨 S2 — negative control: the gate before annotation

The failure mode I most expect: the matcher scores "is this stretch in-scale and roughly
descending" and not "is this the phrase". That is invisible if you only look at its top hits.
So, **with zero labels**:

1. Score phrase P (from raag A) against (a) raag A's clips, (b) clips of raags that **share
   A's scale** (`scale_twins.py` next door already computes these), (c) clips of unrelated raags.
2. If (a) does not separate from (b), the matcher has no phrase-specific signal and no amount
   of annotation will rescue it — go back to S1.
3. Report per phrase, by specificity tier. Expect the 2-swar tier to fail this test *by
   construction* — that is the evidence for dropping it, and for telling you which ~90 phrases
   are worth your attention.
4. Second control: a **shuffled-phrase** baseline (same swar multiset, scrambled order). If
   order doesn't matter, we are measuring a histogram.

**Gate: separation on (a) vs (b) for the specific tier. Nothing below this line runs until it passes.**

### 🟥 S3 — annotation, bootstrapped (needs you)

Only after S2. Design choices that matter more than the tool:

- **Verify candidates, don't transcribe from scratch.** Exhaustive labelling of 10 h is out.
- **But verification-only labels measure precision and can never measure recall.** So two pools:
  - **Pool V (verification, ~600 judgments)**: ~20 phrases × ~30 candidates, sampled
    **stratified across score deciles** — not the top-K. Labelling only top hits teaches the
    tuner nothing about where the boundary is.
  - **Pool R (recall, small)**: 3–5 clips swept end-to-end by you for *all* phrases of their
    raag. ~2 h of your time, and it is the only way to know what we are missing.
- **Verdict is 3-way**: `yes` / `no` / `unsure`. (Review: "right notes, wrong feel" is not
  penalised — in professional recordings of the raag, the note combination carries the feel.)
- **You choose the phrases** (review Q6) from `results/phrases.csv` + S1/S2 plots: only ones you
  are confident belong to the raag and know how they sound.
- **First loop is small**: ~150 judgments over ~5 phrases (review Q1).
- Tool: local single-key CLI or notebook — audio snippet (±2 s context) + contour plot with the
  aligned swar path drawn on it + the swar string. Append-only JSONL:
  `{clip_id, t0, t1, phrase, raag, verdict, notes, annotator, ts, score, config_hash}`.
- **Re-show 10% silently** to get your own self-agreement. If you disagree with yourself 20% of
  the time, that caps every number downstream and we should know it.

### 🟥 S4 — tune the heuristic on labels (still no learning)

- Coordinate/grid search over the ~6 costs (ornament cost, gap cost, dwell prior, cents
  tolerance, octave-fold, normalisation) maximising **per-phrase average precision** on Pool V.
- **Folds grouped by video**, train split only. A phrase found in 3 chunks of one recording is
  one observation, not three.
- Report the tuned-vs-default delta, and the loss when `notes-yes-feel-no` is counted as
  positive vs negative — that quantifies how much of the problem the note path solves.

### 🟥 S5 — light sequential learning

Only what the labels can support (hundreds of positives, so tens of parameters):

- **Per-phrase HMM fit**: replace hand-set dwell/ornament costs with ones estimated from
  confirmed positives (dwell distribution per swar, which ornaments actually occur where).
  This is the direct attack on challenge 3 — the phrase's *rendering*, learned.
- **Shared ornament channel**: `motif-classifier/methods/m5_channel.py` already Baum-Welch-fits
  a 12×12 `P(observed | intended)` emission matrix — a model of tracker+ornament behaviour,
  pooled over raags, so it cannot leak raag identity. Port it to frame level and reuse.
- **Reranker**: logistic regression over cheap features (alignment cost, dwell-profile match,
  nyas landing, direction, duration) on the candidates. ~10 weights.
- Grow Pool V by active learning: label the new model's *uncertain* band, not its confident one.

### 🟥 S6 — deep methods

Not until S4/S5 plateau **and** Pool R shows the ceiling is recall, not precision. What would
justify it: a contour encoder trained with the confirmed positives as a metric-learning signal
(the `melody-first/` survey lists the candidates). Written down so we can say no to it on purpose.

---

## Evaluation protocol

Fixed now, so nothing gets chosen after the fact.

- **Unit**: a candidate interval. A hit needs `IoU ≥ 0.5` with a labelled positive — phrase
  boundaries are genuinely fuzzy, so exact endpoints are not the claim.
- **Headline metric**: per-phrase **average precision**, macro-averaged within specificity tier.
  Precision@1 and @5 reported alongside (that is what a user of this actually feels).
- **Recall** only from Pool R. Stated as an estimate with its n, never as "recall".
- **Always alongside**: the S2 null (same phrase, scale-twin raags) and the shuffled-phrase
  baseline. A precision number without them means nothing.
- Train split only, video-grouped, throughout. The test split is not touched by this project
  until there is something finished to measure once.

---

## Decisions from review (2026-09-20)

1. **Annotation budget.** Pool V at ~600 judgments is maybe 2–3 h. Pool R is ~2 h more. Is that
   the right size, or should I aim smaller for a first loop (say 150 judgments over 5 phrases)?
   <br>--> Smaller first loop please.
2. **Which raags first?** I'd pick 5 with distinctive long phrases and clean audio — Bageshree,
   Malkauns, Bhoopali, Darbari, Lalit — rather than sampling all 50 thinly. Objection?
   <br>--> Sounds good -- select raags that have more distinctive phrases. I suggest: Bageshree, Shree, Puriya Dhanashri, Malhar, ... although open to more/others, not just 5.
3. **`notes-yes-feel-no`**: is that the right third category, or do you want to split it
   (wrong tempo / wrong ornament / wrong emphasis)? Your call — you're the annotator.
   <br>--> Hmm this one is tricky. For now, I'll refrain from penalizing this -- if the note combo exists in the given raag, in these professional musician recordings, it probably fits the feel bill.
4. **Phrase tiering**: confirm we drop the 33 two-swar entries and the 65 that occur in ≥10
   raags, i.e. work the ~90 unique ones. Or are some 2-swar entries real pakads to you
   (`Bageshree: ,n ,D` reads like one) that I shouldn't discard?
   <br>--> Yes, please discard these. You may also place weightage on more idiosyncratic phrases, more complex phrases (e.g. "G m P" is commonly found but "G D P" is more special though the length is the same)
5. Clips are 20 s chunks of longer videos. `../raag-identifier/hindustani-raag-fullaudios/`
   exists — worth using full recordings later, or stay on the pinned dataset? (Staying, unless
   you say otherwise; the pin is in CLAUDE.md.)
   <br>--> For exploration and on an as-needed basis, we can use the full audios. Prefer the pinned for all
   reproducible workflows. If that's not enough, though, open to switching to full audios too.
6. **Choosing phrases for annotation** I'd like to select which ones to annotate here. My own
   musical (incomplete) training v/s the source of mukhyangas in the DB may be have divergences; I'll
   stick to phrases I'm confident about belonging to a raag + knowing how they feel. Also note that the
   database phrases are often a suggestion and not an exhaustive holy grail or an airtight raag signature.
   Our goal is to do our best at identifying where they lie, if they do.

---

## Reuse map

| from | what | note |
|---|---|---|
| `../raag-identifier/utils/config.py` | dataset/cache paths, pinned revisions | as-is |
| `../raag-identifier/utils/dataset.py` | `load_clips`, tonics | as-is |
| `../raag-identifier/utils/raagdb.py` | `Raag`, `parse_phrase`, `collapse`, `ngram_document_frequency` | as-is; phrase tiering built on the last |
| `../raag-identifier/utils/extract.py` | `_essentia` tracker | call it; **new cache, f0 only** |
| `../raag-identifier/melody-extraction/note_segmentation.py` | — | **deliberately not used**, see above |
| `../raag-identifier/motif-classifier/methods/m5_channel.py` | Baum-Welch ornament emission matrix | port in S5 |
| `../raag-identifier/motif-classifier/scale_twins.py` | scale-twin raag pairs | the S2 null set |

Nothing outside `../raag-identifier/` is imported.

---

## Log

- **2026-09-20** — Read the problem. Probed the data (throwaway scripts, not checked in):
  confirmed annotated tonics put notes on the right swar grid (12/12 clips, k=0);
  found that the shared note segmentation's short-segment merge *lowers* duration-weighted
  in-scale (0.67 vs 0.90) and that no verbatim `m D n D` survives in 5 Bageshree clips.
  Both push the whole project onto the frame-level contour rather than a symbol string.
  Plan written; reviewed (answers inline above).
- **2026-09-20** — S0 ✅ (f0 cache, phrase catalogue: 159/229 kept). S1 ✅: matcher + plots over
  8 focus raags (Bageshree, Shree, PuriyaDhanashri, Malhar, Malkauns, DarbariKanada, Lalit,
  Bhoopali). Four scoring revisions, all driven by plots (table in S1). Plots restyled per
  review. **Next: S2** — per-phrase null (scale-twins, unrelated raags, shuffled phrase).
