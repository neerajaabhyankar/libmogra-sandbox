# survey-aug-2026 — DL raag classification on dataset v1.1

Lab notebook. Newest findings go at the bottom of each stage; **nothing is deleted, including
things that failed** — a negative result costs the same to produce as a positive one and is
worth more than a blank space.

Standing brief and rules: [`CLAUDE.md`](CLAUDE.md). Run registry: [`RUNS.md`](RUNS.md).

**Every results table here is generated, never typed**, so the entries read as one sequence
and no cell can drift from what was measured:

```bash
poetry run python scripts/90_report.py --notebook c1 c2 c2_shuffled   # paste the output
```

One shape throughout — run, what, val top-1, test top-1, vs that architecture's Stage 1,
mistake affinity against its chance floor. Put the run being tested next to its comparator
in the id list and the reader can see the delta without arithmetic. Column specs live in
[`common/report.py`](common/report.py), which is also what writes `RESULTS.md` and what
`scripts/status.sh` prints — one loader, one field-deriver, one renderer.

---

## The bar

| where | method | what it is | v1.1 test top-1 |
|---|---|---|---|
| `../motif-classifier` | **M9+** | melody surface + phrase grammar, no learned audio model | **0.400** (60/150) |
| `../motif-classifier` | **M12** | pitch histogram + libmogra DB prior, chi² | **0.400** (60/150) |
| `../motif-classifier` | M14 | M12 + bigram LM — best CV (0.468), worst gap (+0.095) | 0.373 |
| `../motif-classifier` | M11 | pitch histogram, *no musical knowledge at all* | 0.387 |
| `../hindustani-raag-classifier-resnet` | Stage 6a | jeevster ResNet, last 2 blocks unfrozen — **on v0** | 0.174 (16/92) |
| `../distilhubert-finetuned-hindustani-raag-small` | — | distilHuBERT fine-tune — **on v0** | (see below) |

Chance = 0.020. **The number to beat is 0.400**, and the uncomfortable fact framing this whole
folder is that a 120-bin pitch histogram with no learning (M11, 0.387) is currently within
0.013 of the best method in the project, while the best supervised deep model in the repo
managed 0.174 — on an easier split.

Two caveats before treating that 0.174 as the DL baseline:

- It was measured on **v0** (1161 train / 92 test, ~6 s clips, *estimated* tonic), not v1.1.
- Neither old DL run **grouped its splits by video**. `hrs.train_test_split(stratify_by_column=
  "label")` in the distilHuBERT notebook shuffles clips, and three chunks of one recording land
  across train and val. Its val accuracy therefore measured recording recall, not raag
  recognition. Every split in this folder is grouped by video id.

---

## The dataset, as it actually is

`neerajaabhyankar/hindustani-raag-small@326caef0bc01da44ad46e4d9c65a5146da6bcc5b` (v1.1),
already materialised at `../hindustani-raag-small-v1.1/` by `utils.dataset.fetch`
(audio symlinked to `../hindustani-raag-small-v1/` — byte-identical, only `tonic_hz` moved).

| | |
|---|---|
| clips | 1960 — **1810 train / 150 test** |
| raags | 50, all 50 present in test at exactly **3 clips each** |
| videos | 412; clips come in ~3-chunk groups from one recording |
| duration | 20 s for all but a handful (max 41 s) |
| per-raag clips | 18 … 73 (unbalanced) |
| `tonic_hz` | **hand-annotated per video**, 101 – 289 Hz, median 147 |

The v1.1 tonic is the thing v0 did not have, and `../motif-classifier/plan.md` is unambiguous
about what it is worth: with an *oracle* tonic, its M3 went 0.109 → 0.314 top-1. Sa placement
was worth roughly 3× everything else in that project combined. That is the prior going in.

---

## Design decisions, and why

**Full 20 s clips, not crops.** The old ResNet work trained on 5 s random crops. A raag is
identified by phrase-level movement; 5 s of a 20 s clip is often one or two swars of an alap.
Cropping is a 4× speedup that throws away the evidence, so it is off. Cost measured below.

**Everything grouped by video.** `common/data.py` only exposes video-grouped splits. There is
no code path that produces a clip-level shuffle.

**Checkpoints are selected on val top-1, not val loss.** Found the hard way in the first
`c1` run: a 50-class model's validation cross-entropy bottoms out by epoch 1-2 and then
climbs as the model grows confident, while val top-1 keeps improving for another five or ten
epochs. Selecting on loss picked epoch 1 at top-1 **0.017** when epoch 6 scored **0.070**.
Selecting on the reported metric biases the val column upward slightly; the test column,
scored once, is the unbiased one.

**Test is scored once.** Selection happens on grouped CV / a grouped val split over the 1810
train clips. `common/metrics.py` will refuse to touch the test split unless a script passes
`split="test"` explicitly, and every test evaluation is logged in `RUNS.md`.

**Three architectures.**

| tag | model | tonic enters by |
|---|---|---|
| **D** | distilHuBERT (`ntu-spml/distilhubert`, 23.7 M, 2 layers) on 16 kHz waveform | pitch-shifting the waveform, or FiLM conditioning |
| **R** | jeevster 1D-ResNet (5.5 M) on 8 kHz stereo waveform | same |
| **C** | small 2D ResNet on **CQT rolled so Sa = bin 0** | *structurally* — the roll makes it exact |

C is the addition to the brief. The argument for it: for a waveform model the tonic can only
be injected as a nuisance parameter the network must learn to use, whereas rolling a CQT makes
the representation **exactly tonic-invariant by construction**, at zero cost. It also lands the
model's features in the same 12-bin swar-occupancy space that the libmogra templates live in,
which turns the DB-prior experiment from a bolt-on into a natural output layer.

**Two ways to give a waveform model the tonic, both to be tried:**

- *Normalise* — resample the audio by `2^(-k/12)` where `k` = the octave-folded distance from
  the annotated Sa to a fixed reference. Octave folding keeps the ratio inside ±6 semitones,
  so the tempo distortion stays under 1.41×. Every clip then has Sa in the same place, and the
  model never has to learn the transposition.
- *Condition* — leave the audio alone, feed `log2(tonic_hz)` (and its 12-way pitch class,
  one-hot) into the head via FiLM. Cheaper, no distortion, but the model must learn to use it.

**The control that catches the bug.** Every tonic experiment is run a third time with the
tonics **shuffled between videos**. If the shuffled run matches the real one, the tonic is not
reaching the model and the result is plumbing, not music.

---

## Measured cost (Apple M1, 17 GB, MPS)

Benchmarked before writing any training code, because it decides what is a local run and what
is an overnight script.

| model | input | batch | s/step | min/epoch (1810 clips) |
|---|---|---|---|---|
| distilHuBERT, **feature encoder unfrozen** | 20 s | 8 | **87.1** | 328 — memory thrash, unusable |
| distilHuBERT, feature encoder frozen | 20 s | 2 | 0.78 | 11.8 |
| distilHuBERT, feature encoder frozen | 20 s | 4 | 1.46 | 11.0 |
| distilHuBERT, feature encoder frozen | 20 s | 8 | 3.13 | 11.8 |
| distilHuBERT, feature encoder frozen | 10 s | 8 | 1.26 | 4.7 |
| distilHuBERT, feature encoder frozen | 5 s | 8 | 0.56 | 2.1 |

Reading: **freezing the convolutional feature encoder is not a modelling choice here, it is the
difference between 12 minutes and 5.5 hours per epoch.** It is also standard practice for
HuBERT fine-tuning, so no loss. At ~12 min/epoch a 20-epoch distilHuBERT run is ~4 h — an
overnight `scripts/` job, and a 5-fold grouped CV of it is not affordable locally.

---

## Stages and batches

**Stages** are the questions, from the brief. **Batches** are how they were run — a batch may
cover several stages, and a stage may span batches.

| stage | question | settled in |
|---|---|---|
| Stage 0 [harness] | do the splits, metrics and caches work; is there signal at all | Batch 0 |
| Stage 1 [baseline] | retrain each architecture as-is on v1.1 | Batches 1, 2 |
| Stage 2 [tonic] | does the annotated `tonic_hz` help, and how must it enter | Batches 1, 2 |
| Stage 3 [separation] | does source separation as pre-processing help | Batch 3 |
| Stage 4 [DB prior] | does the libmogra database help, and by which mechanism | Batches 3, 4 |
| Stage 5 [hybrid] | do the DL and symbolic methods add up | not yet run |

## Stage detail

Status legend: ☐ not started · ▶ running · ✔ done · ✖ tried and failed · ⏸ parked

### Stage 0 [harness]
✔ `common/` modules, audio cache (2.34 GB, 0 errors), Sa-anchored and fixed-fmin CQT caches
✔ Frozen-representation probes, video-grouped 5-fold CV — **see the log entry below**

### Stage 1 [baseline]
☐ D1 distilHuBERT, original recipe, v1.1, grouped val  *(batch 2, ~4 h)*
☐ R1 jeevster ResNet, Stage 6a recipe (last 2 blocks + head), v1.1, grouped val
▶ C1 CQT-ResNet with fixed fmin (absolute pitch) — the control for Stage 2

### Stage 2 [tonic]  *(important)*
☐ D2n / R2n — tonic-normalised audio · D2c / R2c — FiLM conditioning
☐ C2 — tonic-rolled CQT
☐ shuffled-tonic control for each

### Stage 3 [separation]
☐ HPSS (`../source-separation`) in front of the best Stage 2 config per architecture.
  Prior from motif-classifier: separation helped the *tracker* and **hurt classification**
  (M9 0.422 → 0.393), the theory being that HPSS smooths away meend and gamak. A spectral
  model may not care about the same thing a 120-bin histogram cares about, so it is worth one
  run per architecture — but the expectation is set low.

### Stage 4 [DB prior]  *(important)*
☐ P1 musically-graded label smoothing from `raagspace.affinity()`
☐ P2 auxiliary head predicting the DB swar-occupancy vector (multi-task)
☐ P3 fixed DB-template output layer (C only — project to occupancy, score by chi²)
☐ P4 inference-time log-prior fusion with M12's scores
  Prior from motif-classifier: the DB is **a good prior and a poor model** — blending at
  λ=0.3 beat both λ=0 and λ=1 for both M12 and M13. Expect the same shape.

### Stage 5 [hybrid]
☐ Only if nothing above beats 0.400. Late fusion of DL probabilities with M9+/M12 scores, or
  M9 melody-surface features concatenated into the DL head.

---

## Log

*(dated entries appended as work happens)*

### 2026-08-31 — folder set up

Established the facts above: dataset shape, the two old DL runs' v0 provenance and their
ungrouped splits, and the cost table. No models trained yet.

### 2026-08-31 — Stage 0: the harness is right, and the probes already say a lot

`scripts/01_probe_representations.py`, video-grouped 5-fold CV over the 1810 train clips.
Every representation is **frozen** — only a linear model is fitted on top — so these are
minutes each and they say what is in the representation before any fine-tuning muddies it.

| representation | classifier | dim | top-1 | top-5 | MRR | macro-F1 | video |
|---|---|---|---|---|---|---|---|
| **melody_hist** | **logreg** | 120 | **0.434** | 0.779 | 0.585 | 0.412 | 0.610 |
| melody_hist | chi² | 120 | 0.406 | 0.737 | 0.551 | 0.392 | 0.599 |
| melody_hist | cosine | 120 | 0.354 | 0.671 | 0.500 | 0.349 | 0.530 |
| jeevster_frozen | logreg | 300 | 0.262 | 0.588 | 0.412 | 0.254 | 0.448 |
| chroma_anchor | logreg | 36 | 0.234 | 0.528 | 0.375 | 0.213 | 0.312 |
| chroma_argmax | logreg | 36 | 0.172 | 0.408 | 0.294 | 0.151 | 0.238 |
| chroma_anchor | chi² | 36 | 0.165 | 0.402 | 0.288 | 0.157 | 0.243 |
| chroma_anchor | cosine | 36 | 0.142 | 0.366 | 0.262 | 0.135 | 0.221 |
| chroma_argmax | chi² | 36 | 0.130 | 0.380 | 0.260 | 0.116 | 0.174 |
| **chroma_fixed** | logreg | 36 | **0.067** | 0.209 | 0.157 | 0.060 | 0.094 |
| **hubert_frozen** | logreg | 768 | **0.064** | 0.147 | 0.129 | 0.065 | 0.069 |
| chroma_fixed | chi² | 36 | 0.043 | 0.149 | 0.123 | 0.039 | 0.044 |

**1. The harness is correct.** `melody_hist / chi²` is M11 rebuilt here — the same CREPE
tracks from `../motif-classifier/cache/`, the same `fold_histogram`, with only this
project's splits, label mapping, tonic lookup and metrics in between. It scores **0.406**
against M11's **0.395** over there. The difference is config detail (M11 tunes `smooth` and
`power`); the agreement is the point. Splits, labels, tonic and metrics are all wired right,
so a bad number later is a bad model and not a bad harness.

**2. Sa-anchoring the CQT is worth 3.5×, measured with nothing trained.** `chroma_anchor`
and `chroma_fixed` are the *same CQT* of the *same audio*, differing only in whether `fmin`
is the clip's own tonic or a constant 55 Hz:

| | chi² | logreg |
|---|---|---|
| fixed fmin (absolute pitch) | 0.043 | 0.067 |
| **fmin = Sa** | **0.165** | **0.234** |

A fixed-fmin pitch profile is barely above chance, because the corpus spans an octave and a
half of tonics and the same raag lands in a different place in every recording. This is the
cleanest statement of the tonic lever anywhere in the project: it costs nothing, no model
sees it, and it is a 3.5× multiplier. It is also the argument for the C architecture,
now measured rather than asserted.

**3. Off-the-shelf distilHuBERT embeddings are close to useless here — 0.064.** Three times
chance from a 768-dim representation, worse than a 36-bin pitch histogram. That is not
surprising for a speech model distilled for phonetic content, but it sets expectations for
Stage 1: **everything D achieves will have to come from fine-tuning**, and there is no
warm-start advantage to lean on. It also makes the ~4 h/run price look worse.

**4. The Carnatic ResNet transfers, and better than the sibling project managed.**
`jeevster_frozen` + a linear probe scores **0.262** — 13× chance, from a backbone trained on
different repertoire and never shown a Hindustani label. For scale, the sibling project's
frozen-backbone-plus-new-head model scored 0.109 on v0 test. Much of that gap is v0 vs v1.1
and the ungrouped split, but the direction is encouraging for R.

**5. `logreg` > chi² > cosine, on every representation.** ../motif-classifier established
chi² > cosine and made it a project-wide rule. Extending the ladder one step: a
**discriminative** linear model beats template matching on the identical feature, by +0.028
on melody_hist (0.434 vs 0.406) and by +0.069 on chroma_anchor (0.234 vs 0.165). That is not
a small effect, and it is the one thing the symbolic project never tried — every method
there scores a clip against a per-raag *template*, generatively. **A plain multinomial
logistic regression on M11's own 120-bin histogram (0.434) beats M11 (0.395), M12 (0.412) and
every other CV number in that project except M14 (0.468) and M9+ (0.447).** Worth reporting
back to the motif-classifier line of work independently of anything here.

**6. Negative result: `chroma_argmax` is worse than `chroma_anchor`** (0.172 vs 0.234). The
idea was that taking only the loudest CQT bin per frame approximates a melody line and dodges
harmonic contamination — a note's 3rd harmonic lands on its fifth, its 5th on its major
third. It does the opposite: throwing away all but one bin per frame loses more than the
harmonics cost. Full-energy chroma stays the default.

**And the gap that frames the DL work:** a 120-bin histogram of a pitch track scores 0.434,
while the best *spectral* representation here scores 0.234. Whatever the deep models are
going to contribute, the melody-extraction pipeline is currently a much better front end
than the raw spectrum, and Stage 5's hybrid is looking less like a stretch goal and more
like the obvious thing to do.

### 2026-08-31 — Stages 1 and 2 on the cheap architectures: the tonic must be *in* the representation

Batch 1, all six runs. Single grouped val split (1350 fit / 460 select, disjoint videos),
checkpoint on val top-1. Chance is 0.020.

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| c2 | CQT, Sa-anchored | 0.302 | 0.187 | +0.191 | 0.380 (0.263) |
| r2n | ResNet, tonic-normalised audio | 0.287 | 0.227 | +0.141 | 0.363 (0.262) |
| r1 | jeevster ResNet, as-is | 0.146 | 0.113 | — | 0.305 (0.260) |
| c1 | CQT, fixed fmin | 0.111 | 0.080 | — | 0.297 (0.259) |
| c2_shuffled | c2, tonics permuted *(control)* | 0.087 | 0.040 | -0.024 | 0.275 (0.259) |
| r2c | ResNet, tonic by FiLM | 0.057 | 0.033 | -0.089 | 0.250 (0.260) |

**The tonic is worth 2-2.7x, and the shuffled control now proves it is the tonic.**
`c2_shuffled` anchors each video's CQT at a deliberately wrong Sa and collapses to 0.087 —
*below* c1's 0.111, which is what it should be: a wrong anchor is worse than no anchor,
because it scatters the same raag across different bin offsets depending on which video a
clip came from. c2 beats its own control by 3.5x.

The graded metrics say the same thing in a way accuracy cannot. c2's mistakes land at
affinity **0.380** against a chance of 0.263 — when it is wrong, it is wrong about a
musically adjacent raag. c2_shuffled's mistakes sit at 0.275 against chance 0.259, i.e.
indistinguishable from guessing. The shuffled model has not learned a blurry version of
the structure; it has not learned the structure.

Read the first version of this control with suspicion: it originally scored 0.313 and
appeared to show the tonic did not matter. That was a cache-key bug
(`cached_cqt` keyed on the string `"anchor"` rather than the anchor frequency, so the
shuffled run silently reused the correctly-anchored CQTs). Fixed, control re-run. The
lesson is cheap to state and was expensive to catch: **a control that fails to separate is
first evidence about the harness, not about the hypothesis.**

**FiLM conditioning does not work here — and this one is not yet trustworthy.** r2c feeds
the tonic as a conditioning vector into FiLM layers instead of normalising the audio, and
scores 0.057 against r1's 0.146. Adding information made it substantially worse, and it
early-stopped at epoch 4. Two readings:

1. Real. A 13-d conditioning vector against a 300-channel backbone is a weak signal that
   mostly perturbs the features, and normalising the audio hands the model the same
   information for free.
2. A bug, exactly like c2_shuffled's. The FiLM path may not be wired to anything that
   matters, or may be scrambling the features.

**Do not report the FiLM result until `r2c_shuffled` has run.** If shuffling the
conditioning tonic leaves the score unchanged, the tonic never reached the FiLM layers and
0.057 means nothing. That is the same diagnostic that just caught the CQT cache bug, and
it costs 28 minutes.

**Where this leaves the bar.** c2's 0.302 is val, and motif-classifier's 0.400 is test, so
they are not directly comparable — but the DL side is now within reach of the symbolic
champion rather than a factor of three behind it, and Stages 3 and 4 are untouched.

**Stage 3 and 4 configuration is now decided**: both go on top of *audio-level* tonic
normalisation (`--tonic normalise`), for both architectures. FiLM conditioning is parked
pending its control.

### 2026-08-31 — Stages 1 and 2 on distilHuBERT: a dead end, and the tonic cannot rescue it

Batch 2, three runs on a Colab T4 (`device=cuda`, batch 16), same pinned revision and same
video-grouped split as Batch 1.

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| d1 | distilHuBERT, notebook recipe | 0.080 | 0.053 | — | 0.266 (0.260) |
| d2n | distilHuBERT, tonic-normalised audio | 0.087 | 0.060 | +0.007 | 0.285 (0.260) |
| d2c | distilHuBERT, tonic by FiLM | 0.076 | 0.047 | -0.004 | 0.267 (0.260) |

**Fine-tuning distilHuBERT bought almost nothing over not fine-tuning it.** Frozen
distilHuBERT embeddings scored 0.064 in the Stage 0 probe. Twenty epochs x 3 runs, ~3 hours
of T4, moved that to 0.080. Against 0.302 for the CQT net, which trains in 17 minutes on a
laptop.

**The tonic does nothing here, and that is the interesting part.** It was worth 2.0-2.7x on
both other architectures. On distilHuBERT the three conditions span 0.076 to 0.087 — noise.

A tonic-invariant result is precisely the silhouette of the `c2_shuffled` cache bug, so this
was checked before being believed. It is not that bug: `cached_waveform` stores only decoded
audio, and the pitch shift is applied downstream in `clip_tensor`, never cached, so there is
no key to collide on. Directly verified — the d1 and d2n tensors for the same clip correlate
at **0.009**. The model is seeing genuinely different audio and scoring the same on it.

The graded metrics say why. All three sit at mistake affinity 0.266-0.285 against a chance
floor of 0.260: **the mistakes are musically random.** These models have not learned raag
structure that a correct tonic could sharpen. There is nothing for the tonic to align, so
supplying it — by either route — changes nothing. Compare c2, whose mistakes land at 0.380
against the same floor.

This is the cleanest use the graded metrics have had so far. Top-1 alone would say
"distilHuBERT is weak"; mistake affinity says "distilHuBERT has learned nothing musical",
which is a different claim and the one that justifies stopping.

**Decision: distilHuBERT is parked.** A speech-pretrained model at 16 kHz appears not to
represent pitch-relative structure at the resolution this task needs, and it costs ~55
min/run against the CQT net's 17. One exception is worth a single run: the convolutional
feature encoder was frozen throughout (87 s/step unfrozen on the M1, laptop-infeasible), so
`d1_unfrozen` is the one untested explanation. It is one cell in the Colab notebook.

**Where the survey stands after Stages 1-2** (val top-1, single grouped split):

| | Stage 1 | Stage 2, best |
|---|---|---|
| CQT-ResNet (C) | 0.111 | **0.302** (Sa-anchored) |
| jeevster ResNet (R) | 0.146 | 0.287 (normalised audio) |
| distilHuBERT (D) | 0.080 | 0.087 (no real effect) |

Two of three architectures respond strongly to the tonic, and only when it is baked into
the representation rather than supplied as a side channel. Stages 3 and 4 proceed on C and
R only.

### 2026-08-31 — Stages 3 and 4 launched

Batch 3, six runs, all on top of Stage 2's winner (the tonic at the audio/representation
level). FiLM conditioning is not used by any of them; Batch 1 ruled it out.

**Stage 3 — source separation (c3, r3).** Both over the HPSS melody stem. Going in with a
strong prior *against*: motif-classifier ran exactly this experiment for the symbolic
methods and every method that saw separated audio got worse — M9 -0.028, M12 -0.012,
M14 -0.026, with M13 (which reads un-re-extracted notes) flat to three decimals as the
control that says those deltas are real. The reading there was that HPSS smooths away meend
and gamak along with the tabla, and a 120-bin histogram is built specifically to see
sub-semitone movement.

The open question these two runs answer is whether that generalises to a model reading a
**spectrogram** rather than a pitch track. A CQT net is not obviously as dependent on
continuous ornamentation as a histogram is; it may prefer the cleaner harmonic structure. If
it is hurt the same way, that is a second independent confirmation that the ornamentation is
signal, not noise, and separation can be closed out for this dataset.

**Stage 4 — the database as a prior (c4g, c4a, c4h, r4g).** Three mechanisms, cheapest
first, so a null result is diagnosable rather than just disappointing:

- **c4g / r4g, graded label smoothing** (`--graded-alpha 0.3`). Touches the loss only,
  nothing structural. The cleanest read on "does DB adjacency information help at all". On
  both architectures, so a result is not confounded with the CQT design.
- **c4a, auxiliary swar-occupancy head** (`--aux-weight 0.3`). The DB as a second task
  rather than a softer target.
- **c4h, the DB-template head.** Predict a 12-bin swar profile, score it against the
  libmogra templates by chi-square. This is M12's mechanism learned end to end, and it is
  the reason the C architecture was specified with a 12-bin feature space in the first
  place — the templates and the model's features live in the same vector space, so the
  prior plugs in natively instead of being bolted on. If any Stage 4 run justifies the
  architecture choice, it is this one.

Ordered cheapest-mechanism-first on purpose: if c4g moves nothing, that is evidence the DB
adds no information *this model cannot already infer from the audio*, which reframes c4h's
result before it is run.

### 2026-08-31 — the FiLM control: conditioning is wired, and it actively hurts

`r2c_shuffled` permutes the conditioning tonic between videos. Against r2c's 0.057 it
scores **0.130** — the score moved, and moved a lot, so the FiLM path is wired and r2c's
number is a real result rather than the c2_shuffled failure repeating.

The direction is the surprise. Lining up the four resnet1d runs:

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| r2n | ResNet, tonic-normalised audio | 0.287 | 0.227 | +0.141 | 0.363 (0.262) |
| r1 | jeevster ResNet, as-is | 0.146 | 0.113 | — | 0.305 (0.260) |
| r2c_shuffled | r2c, tonic permuted *(control)* | 0.130 | 0.147 | -0.015 | 0.322 (0.260) |
| r2c | ResNet, tonic by FiLM | 0.057 | 0.033 | -0.089 | 0.250 (0.260) |

**Supplying the true tonic through FiLM is worse than supplying a false one, and both are
worse than saying nothing at all.** A correct, informative input made the model
substantially worse — 0.057 against 0.146 — which is not what "the conditioning is too weak
to help" would look like; that would land near r1.

The likeliest mechanism is a shortcut. Tonic is constant per video, so the conditioning
vector is close to a video fingerprint. With grouped splits the val videos are unseen, so
any capacity spent keying on that fingerprint is capacity wasted, and the fingerprint is
most learnable when the tonic is *true* — genuinely correlated with the recording's
acoustics.
Shuffling decorrelates the vector from the audio, the model leans on it less, and the score
drifts back toward r1. That story predicts r2c should also show the largest train/val gap of
the four; worth checking against `history.json` before it is believed.

**Stage 2 conclusion, for both architectures: put the tonic in the representation, never in
a side channel.** c2 (0.302) and r2n (0.287) both normalise; every conditioning variant —
r2c 0.057, d2c 0.076 — is at or below its own no-tonic baseline. Stages 3 and 4 use
`--tonic normalise` exclusively, and `--tonic-mode condition` is retired.

### 2026-08-31 — Stage 3: source separation does not help here either

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| c3 | c2 + HPSS melody stem | 0.304 | 0.220 | +0.193 | 0.381 (0.263) |
| c2 | CQT, Sa-anchored | 0.302 | 0.187 | +0.191 | 0.380 (0.263) |
| r3 | r2n + HPSS melody stem | 0.113 | 0.067 | -0.033 | 0.282 (0.260) |
| r2n | ResNet, tonic-normalised audio | 0.287 | 0.227 | +0.141 | 0.363 (0.262) |

**Closed as a negative.** motif-classifier found every symbolic method got slightly worse on
HPSS audio and read it as HPSS smoothing away meend and gamak along with the tabla. The open
question was whether a model reading a *spectrogram* rather than a pitch track would be hurt
the same way. Answer: the CQT net is indifferent (+0.002, noise) and the waveform ResNet is
severely hurt (-0.174).

That the ResNet suffers most is consistent with the ornamentation story — it reads the raw
waveform at 8 kHz, so HPSS's median filtering removes signal it was using directly, while the
CQT net sees a log-magnitude spectrogram where the harmonic content survives the filtering.
Either way no architecture benefits, and this closes Stage 3 for this dataset. Revisiting
needs a separator that removes percussion without smoothing the melodic line, not more HPSS
tuning.

### 2026-08-31 — Stage 4: the DB-template head is the biggest single win in the survey

**c4h = 0.417**, against c2's 0.302 — **+0.115** from replacing `Linear(D, 50)` with a head
that predicts a 12-bin swar profile and scores it against the libmogra templates by
chi-square. That is M12's mechanism, learned end to end.

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| c2 | CQT, Sa-anchored | 0.302 | 0.187 | +0.191 | 0.380 (0.263) |
| c4h | c2 + **DB-template head** | 0.417 | 0.387 | +0.307 | 0.430 (0.262) |

Everything moves together, which is what a real gain looks like rather than a metric
artefact. Mistake affinity 0.430 is the highest of any run in the survey: when c4h is wrong
it is wrong about a musically adjacent raag more often than any other model here.

**This is the run that justifies the C architecture.** The whole argument for a CQT anchored
so bin 0 is Sa was that its feature space is the same 12-bin space as the libmogra
templates, so the DB prior plugs in natively rather than being bolted on. That was a design
bet made in Stage 0; c4h is the evidence for it.

Not yet comparable to the bar: **0.417 is val, motif-classifier's 0.400 is test.** The test
split has still not been touched. The comparison happens once, at the end.

The three loss-level Stage 4 runs (c4g graded smoothing, c4a auxiliary occupancy head, r4g
graded smoothing on the ResNet) crashed on a harness bug and are re-running; their results
decide whether the DB helps *only* through the structural route or also as a softer target.

### 2026-08-31 — a harness bug that only fired on unused branches

c4g, c4a and r4g all died at the end of epoch 0. One root cause: `trainer.evaluate` computed
the validation loss by rebuilding a dict holding **only** logits, moved to CPU. So

* a graded target matrix `Q` built on `mps` met CPU logits → device mismatch (c4g, r4g);
* the auxiliary occupancy output was absent from that dict → `KeyError` (c4a).

The models were built correctly; evaluation was misrepresenting what the forward pass
produced. `evaluate` now computes the loss from the real forward output, batch by batch, on
device, and `Objective` moves its targets to the logits' device instead of assuming one.

Third bug of this shape in one day, after the CQT cache key and the self-matching process
wait. All three were code that worked on the path being exercised and silently did the wrong
thing on a branch nobody had run yet, and all three surfaced as a *plausible-looking result*
or a hang rather than an error. Cheap rule going forward: **any experiment whose flag has
never been exercised gets a one-epoch smoke run before it is queued behind hours of work.**

### 2026-09-01 — test scores for every method, and Stage 4's real shape

Protocol changed at the user's request: every run now scores the held-out 150 at the end,
and `scripts/91_score_test.py` backfilled the runs that finished before the change (from
their saved `best.pt` — inference only, no retraining).

**The split discipline that makes this safe was verified, not assumed.** Videos are dealt
whole into folds, never clips: fit ∩ val videos = 0 (270 / 92), train-pool ∩ test videos = 0
(362 / 50), and all ten pairwise video overlaps across the 5 CV folds = 0. `c2_shuffled` is
independent evidence for the same thing — a per-video tonic is a video fingerprint, so if
videos leaked the shuffled control would have scored *well*; it collapsed to 0.087 instead.

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| c4h | c2 + **DB-template head** | 0.417 | 0.387 | +0.307 | 0.430 (0.262) |
| r4g | r2n + graded label smoothing | 0.309 | 0.213 | +0.163 | 0.369 (0.262) |
| c3 | c2 + HPSS melody stem | 0.304 | 0.220 | +0.193 | 0.381 (0.263) |
| c4a | c2 + auxiliary occupancy head | 0.302 | 0.200 | +0.191 | 0.380 (0.263) |
| c2 | CQT, Sa-anchored | 0.302 | 0.187 | +0.191 | 0.380 (0.263) |
| c4g | c2 + graded label smoothing | 0.293 | 0.233 | +0.183 | 0.434 (0.262) |
| r2n | ResNet, tonic-normalised audio | 0.287 | 0.227 | +0.141 | 0.363 (0.262) |
| r1 | jeevster ResNet, as-is | 0.146 | 0.113 | — | 0.305 (0.260) |
| r3 | r2n + HPSS melody stem | 0.113 | 0.067 | -0.033 | 0.282 (0.260) |
| c1 | CQT, fixed fmin | 0.111 | 0.080 | — | 0.297 (0.259) |
| d2n | distilHuBERT, tonic-normalised audio | 0.087 | 0.060 | +0.007 | 0.285 (0.260) |
| d1 | distilHuBERT, notebook recipe | 0.080 | 0.053 | — | 0.266 (0.260) |
| d2c | distilHuBERT, tonic by FiLM | 0.076 | 0.047 | -0.004 | 0.267 (0.260) |
| c2_shuffled | c2, tonics permuted *(control)* | 0.087 | 0.040 | -0.024 | 0.275 (0.259) |
| r2c | ResNet, tonic by FiLM | 0.057 | 0.033 | -0.089 | 0.250 (0.260) |

**Every val-only conclusion survives test.** c4h leads both splits by a wide margin (+0.10
val, +0.15 test over the next method); the controls stay pinned near the 0.020 floor; the
architecture ordering C > R >> D is unchanged.

**c4h also generalises best**, and that is the new information. Its val->test gap is +0.031
against +0.116 for c2, +0.102 for c4a, +0.096 for r4g. A model constrained to score a
predicted swar profile against fixed musical templates has far less freedom to fit
video-specific quirks than one free-fitting 50 output classes, and the gap column is what
that looks like empirically.

**Stage 4's shape, now complete.** Four mechanisms, three of which do nothing:

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| c4h | c2 + **DB-template head** | 0.417 | 0.387 | +0.307 | 0.430 (0.262) |
| c4a | c2 + auxiliary occupancy head | 0.302 | 0.200 | +0.191 | 0.380 (0.263) |
| c4g | c2 + graded label smoothing | 0.293 | 0.233 | +0.183 | 0.434 (0.262) |
| r4g | r2n + graded label smoothing | 0.309 | 0.213 | +0.163 | 0.369 (0.262) |

So the useful claim is narrower than "the database helps". The DB's *adjacency* information
— which raags neighbour which — adds essentially nothing, which is consistent with c2
already reaching mistake affinity 0.380 without any DB input: the model had largely worked
that out from audio. What the database contributes is its **swar templates as a scoring
target**. Same information source, and it is worth +0.115 or +0.000 depending entirely on
whether it enters the architecture or the loss.

A detail worth keeping: c4g has the *highest* mistake affinity of any run (0.434, above
c4h's 0.430) while scoring lower on top-1. Graded smoothing did exactly what it promises —
moved errors closer musically — without converting that into correct answers. A method can
be more musical and less accurate at once, which is the whole reason both metrics are
reported.

**Against the bar: 0.387 test, versus motif-classifier's 0.400.** The DL survey does not
beat the symbolic champion. It lands 1.3 points short with a model that never runs a pitch
tracker, and the two are close enough on 150 clips (SE ~4 points) to be a statistical tie.

**On reading this table.** Each test number is honest — training never saw test, selection
was on val. The *maximum* of sixteen test numbers is not: it carries roughly the +4-point
optimism of a best-of-N. c4h is the one method that was also best on val, chosen before its
test score existed, so quoting 0.387 for it is legitimate in a way that quoting the max of
this column would not be.

---

## Next steps

Ordered by expected value per hour, not by ambition. Costs are M1 wall-clock.

### 1. Stage 5 [hybrid] — fuse with the symbolic side  *(~1 h, no training)*

The brief listed a hybrid as a stretch goal *"only if nothing here beats motif-classifier"*.
Nothing did — c4h 0.387 test against M14's 0.400 — so the condition has triggered.

Start with the free version: **average the logits of c4h and M14** over the test clips. No
training, both models exist, both already write logits to disk. The case for expecting a
gain is that they cannot be making the same mistakes: c4h reads a Sa-anchored spectrogram
and never sees a pitch track; M14 reads CREPE notes and never sees spectral energy. Their
mistake-affinity profiles differ too. If fusion moves nothing, that is itself informative —
it would mean the two are exploiting the same underlying cue by different routes.

Then the trained version: concatenate M11's 120-bin melody histogram onto c4h's pooled
feature before the DB head. Stage 0 already showed that histogram carries 0.434 under a
plain logistic regression, higher than anything the DL side reached — it is the strongest
single feature in the project and no neural run has been given it.

### 2. Stage 4 [DB prior] — push the mechanism that won  *(~2 h — in Batch 4)*

`--db-bins` has choices 12 / 36 / 144 and **has only ever run at 12**. `--db-lam` has only
ever run at 0.3, inherited from M12's optimum for a different model. Neither was tuned; c4h
is the first point sampled, not the best one found.

| run | change | what it settles |
|---|---|---|
| c4h_36 | `--db-bins 36` | 3 bins/semitone ≈ 33 cents. Stage 3 showed sub-semitone movement is *signal* — a 12-bin profile quantises meend and gamak away, which is precisely the resolution this repo exists to study |
| c4h_144 | `--db-bins 144` | the CQT's own resolution, folded; matches `dbprior.pitch_template` |
| c4h_lam0 / lam1 | `--db-lam 0 / 1` | how much of the win is the *database* versus the *shape of the head*. lam=0 is learned templates with the same architecture — the honest ablation, and it is missing |
| c4h_frozen | `--db-lam 1 --db-freeze-templates` | 50 scalar biases as the only raag-specific parameters. If this holds up, the model is a shruti-profile estimator and the DB does the classifying |

`c4h_lam0` is the one I would run first: without it, "the DB prior is worth +0.115" is not
yet separable from "a template-scoring head is worth +0.115".

### 3. A temporal swar head — the actual new architecture  *(~1 day, not yet queued)*

**Evidence it is the right next model.** c4h pools over time before scoring, so it sees a
raag's *pitch histogram* and nothing about order. Its test errors concentrate exactly where
that hurts: **same-scale confusions are 10.9 % of errors against 1.3 % chance, an 8.4×
enrichment**, and the pairs are the textbook ones — Bhupali/Deshkar, Bageshree/Bheempalasi,
Bihag/Hameer. Raags sharing a scale differ in phrase, approach and emphasis, none of which
survives pooling.

Proposed: CQT → CNN → **per-frame 12- or 36-bin swar posterior** (a sequence, not a pooled
vector), then two heads over that sequence:

* the c4h head unchanged, over the time-averaged posterior — keeps what works;
* a transition/phrase head: bigram statistics of the posterior scored against the DB's
  aaroha, avaroha and mukhyanga, i.e. M13's mechanism learned end to end.

That is M12 + M13, which as a symbolic ensemble was motif-classifier's best method (M14,
0.468 CV). Here the two would share a backbone and train jointly, and the per-frame swar
posterior is independently interpretable — you can plot it against the audio and see whether
the model has found the raag's phrases.

Honest ceiling: fixing *every* same-scale error takes test 0.387 → 0.453. Real, but the
other 89 % of errors are elsewhere, so this is one contribution and not the whole gap.

### 4. Rigour that is currently missing  *(~3 h — in Batch 4)*

Across all 16 runs these never varied: `seed=0`, `folds=1`, `gain_jitter=0`, `freq_jitter=0`,
`length_policy=fixed`, `seconds=20`, `fold_octaves=False`.

* **Seeds.** One seed per configuration. Every ranking in this notebook is a single draw,
  and the selection-optimism analysis showed ±0.05 of peak-picking noise on val. Three seeds
  for c4h and c2 would put an error bar on the headline.
* **5-fold CV for the headline.** `--folds 5` on c4h pools out-of-fold predictions over all
  1810 clips instead of judging on 460. ~100 min, and it is what the final claim deserves.
* **Augmentation was never switched on.** `--freq-jitter` was written specifically for this
  architecture — sub-semitone pitch jitter that models tuning drift between performances
  without moving a swar — and it has run at 0 every time. With ~36 clips per class, this is
  cheap and plausibly worth more than any architectural change above.

### 5. Pretraining on in-domain audio  *(parked — no data on disk yet)*

**Status: not runnable today.** `../data-dunya-hindustani/` does not currently hold the
full corpus, so this is a plan rather than a queued run. Written down because it is the
only idea here that addresses the constraint everything else is bumping into.

**Why it is the right idea.** Batch 2 killed distilHuBERT: three runs, ~3 h of T4, final
score 0.080 against 0.064 for the *frozen, untrained* embeddings. Fine-tuning bought two
points. The natural conclusion is not "pretraining does not work" but "**speech**
pretraining does not work" — a model trained to discriminate phonemes has no reason to
represent pitch-relative structure at the resolution a raag lives in, and its mistake
affinity sat at chance, meaning it never learned musical structure for the tonic to sharpen.

Meanwhile the binding constraint is data: 1810 training clips over 50 classes, ~36 clips per
class. c4h's val→test gap and the ±0.05 of checkpoint-selection noise both trace back to
that. No architecture change fixes a data ceiling; pretraining is the only lever that does.

**The plan, when a corpus exists.**

1. *Corpus.* Unlabelled Hindustani vocal audio — Dunya, or anything with a usable licence.
   Labels are not needed, which is the point; hours matter far more than annotation. A few
   hundred hours would already dwarf the 10 h of labelled clips here.
2. *Representation: the same Sa-anchored CQT the supervised model uses.* This is the crux.
   Anchoring needs a tonic per recording, and unlabelled audio has none — so pretraining
   depends on tonic *estimation* even though the supervised runs use hand annotations.
   Two ways out, and the choice is itself an experiment: estimate a tonic per recording and
   accept the noise, or pretrain on *unanchored* CQTs and let the model learn
   transposition-equivariant features, anchoring only downstream.
3. *Objective: masked-CQT modelling.* Mask spans of time-frequency bins, reconstruct them.
   The Stage 3 result argues for reconstructing at **fine** frequency resolution — 36 bins
   per octave or better — since sub-semitone movement turned out to be signal, not noise,
   and a coarse objective would train the model to discard exactly the meend and gamak that
   distinguish raags.
4. *Transfer.* Replace the CQT-ResNet backbone with the pretrained encoder, keep the
   DB-template head that Batch 3 showed is worth +0.115, fine-tune on the 1810 clips.
5. *The honest control.* A frozen-encoder linear probe, scored against the 0.234 that
   Stage 0's frozen Sa-anchored chroma + logreg already reaches. If pretraining cannot beat
   a 36-bin chroma histogram frozen, it has not learned anything worth the GPU time — the
   same bar distilHuBERT failed at 0.064.

**Cost and honesty about it.** This is days of GPU, not hours, and it is the only item in
this list that could plausibly return nothing. Everything in sections 1-4 should be
exhausted first; they are cheap, and section 1 in particular may close the gap to
motif-classifier without any new model at all.

### 2026-09-01 — Batch 4: the database was not the win, and the seed noise is bigger than the bar

Eight runs, all variations on c4h. Two of them overturn claims made earlier in this notebook.

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| aug_jitter | c4h + pitch/gain jitter | 0.467 | 0.400 | +0.357 | 0.424 (0.266) |
| dbprior_frozen | c4h, database templates frozen | 0.443 | 0.367 | +0.333 | 0.430 (0.263) |
| cv5 | c4h, 5-fold grouped CV | 0.443 | 0.347 | +0.332 | 0.439 (0.266) |
| dbprior_lam0 | c4h, learned templates (`--db-lam 0`) *(ablation)* | 0.435 | 0.373 | +0.324 | 0.407 (0.260) |
| c4h | c2 + **DB-template head** | 0.417 | 0.387 | +0.307 | 0.430 (0.262) |
| dbprior_36bins | c4h at 36 swar bins (~33 cents) | 0.417 | 0.327 | +0.307 | 0.440 (0.264) |
| seed1 | c4h at seed 1 | 0.411 | 0.293 | +0.300 | 0.422 (0.263) |
| dbprior_144bins | c4h at 144 swar bins | 0.404 | 0.347 | +0.293 | 0.435 (0.264) |
| seed2 | c4h at seed 2 | 0.396 | 0.400 | +0.285 | 0.409 (0.264) |

#### The DB-template head works. The database does not.

`--db-lam` interpolates the head's templates from purely learned (0) to the libmogra
database verbatim (1). All else equal:

| lam | templates | val | test |
|---|---|---|---|
| 0.0 | learned from data | 0.435 | 0.373 |
| 0.3 | mostly learned | 0.417 | 0.387 |
| 1.0, frozen | the database, unmodified | 0.443 | 0.367 |

A spread of 0.026 on val and 0.020 on test, against a seed standard deviation of 0.058 on
test. **These three are indistinguishable.** Learned templates do exactly as well as the
libmogra ones, and freezing the database templates so that 50 scalar biases are the only
raag-specific parameters in the model also does exactly as well.

So the earlier claim — *"the DB-template head is worth +0.115; the database contributes its
swar templates as a scoring target"* — was half right and half wrong. The **architecture** is
worth +0.115: predicting a 12-bin swar profile and scoring it against per-raag templates by
chi-square beats a free `Linear(D, 50)` by a wide margin. **Where the templates come from
does not matter.** The constraint is doing the work, not the musicology.

This is the ablation that should have been in Batch 3. It was flagged there as "without it,
+0.115 cannot be attributed to the database" — correctly, and the answer is that it cannot.

Two consolations. It is a *better* result for the architecture: the C design bet was that a
Sa-anchored CQT gives a model a 12-bin space in which raags are separable, and that holds
whether or not a database is consulted. And `dbprior_frozen` says something sharp — a model
whose only raag-specific parameters are 50 biases matches one with a full classifier head, so
the CQT features really are landing in libmogra's own coordinate system.

#### Finer swar bins do not help

36 bins (~33 cents) and 144 bins score 0.417 and 0.404 on val, 0.327 and 0.347 on test — at
or below the 12-bin default. Stage 3 argued sub-semitone movement is signal, and it is, but
apparently not signal a *pooled profile* can use: mistake affinity rises with resolution
(0.430 → 0.440 → 0.435) while accuracy does not. Finer bins make the errors more musical
without making them fewer, which is what you would expect if the extra resolution captures
ornamentation that is real but not raag-discriminative once time order is discarded.

#### The seed spread is larger than every difference this survey has reported on test

c4h, rerun at seeds 1 and 2 (`--seed` also re-deals the grouped split):

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| val | 0.417 | 0.411 | 0.396 | 0.408 | **0.011** |
| test | 0.387 | 0.293 | 0.400 | 0.360 | **0.058** |

Val is stable to ±0.011. **Test moves by 0.107 between seeds.** With 150 clips and one video
per raag, a single unlucky recording swings several clips at once, and the split re-deal
changes what the model saw.

This is the most important result in Batch 4, because it is retrospective. Every test
comparison in this notebook — c4h's 0.387 against motif-classifier's 0.400, c3 against c2,
the val→test gap analysis — was a single draw from a distribution with sd 0.058. None of
those gaps was significant. What survives is the large effects: the tonic (2–2.7×), the
DB-template head (+0.115), distilHuBERT's failure (0.08 against 0.30). Those are multiples of
the noise. Anything under ~0.10 on test should be treated as unresolved.

Going forward: **report test as a mean over three seeds, or do not report a test difference.**

#### Augmentation, never switched on, is the best run in the project

`aug_jitter` — c4h plus `--freq-jitter 2 --gain-jitter 3` — scores **0.467 val / 0.400 test**,
the highest of either column anywhere in this survey, and val is 0.059 above the c4h
seed-mean, five standard deviations of the val noise.

`--freq-jitter` was written for this architecture in Stage 0 — sub-semitone pitch jitter that
models tuning drift between performances without moving a swar to a different swar — and then
ran at 0 in all 16 previous runs. With ~36 clips per class, the cheapest thing in the project
was the one nobody tried.

**It is not yet a claim that the bar is beaten.** 0.400 test equals motif-classifier's 0.400,
on one seed, with test sd 0.058. Three seeds of `aug_jitter` are queued; until they land, the
honest statement is that augmentation is the largest val-side gain since the tonic.

#### `cv5`, the number the headline deserves

0.443 pooled out-of-fold over all 1810 train clips, against 0.417 for the same configuration
on a single 460-clip split. This is the reliable train-side estimate, and it is *higher* than
the single-split value, consistent with the selection-optimism analysis: more folds, less
peak-picking per fold, and five times the evaluation data.

### 2026-09-01 — Batch 5: augmentation was a lucky seed, and the survey ends inside the noise

Two more seeds of `aug_jitter`, the configuration that looked like the best in the project.

| config | val (3 seeds) | mean | test (3 seeds) | mean |
|---|---|---|---|---|
| c4h | 0.417 / 0.411 / 0.396 | 0.408 ± 0.011 | 0.387 / 0.293 / 0.400 | 0.360 ± 0.058 |
| aug_jitter | **0.467** / 0.415 / 0.417 | 0.433 ± 0.030 | 0.400 / 0.340 / 0.380 | 0.373 ± 0.031 |

**Correction to the Batch 4 entry.** It recorded augmentation as "+0.059 val over the c4h
seed-mean, five standard deviations of the val noise". That was seed 0 against a three-seed
mean — the wrong comparison. Seeds 1 and 2 give 0.415 and 0.417, and the honest figures are:

* val **+0.025 ± 0.018** (t = 1.4)
* test **+0.013 ± 0.038** (t = 0.35)

Neither is significant. **Augmentation is not shown to help.** 0.467 was the top of a
distribution, and the mistake was reading a single seed as an effect one entry after writing
down that single seeds cannot be read as effects.

#### Where the survey actually lands

`aug_jitter` test mean **0.373**, 95 % CI **[0.297, 0.449]** on three seeds. motif-classifier's
0.400 sits inside that interval. The correct statement is:

> The best DL configuration scores 0.373 ± 0.031 test top-1 over three seeds. The symbolic
> champion reports 0.400. **The two cannot be separated with this test set.**

Not "the DL side falls 1.3 points short" (Batch 3's phrasing, a single seed against a single
number), and not "augmentation beats the bar" (Batch 4's, the luckiest seed of three).

#### What the test set can and cannot resolve

150 clips, one video per raag. Seed-to-seed sd is 0.031–0.058 depending on configuration, so
the smallest difference this test set can resolve at three seeds is roughly **0.10**. That
covers the survey's real findings — the tonic (2–2.7×), the DB-template head (+0.115),
distilHuBERT's collapse — and none of its fine distinctions.

Everything under 0.10 measured on test in this notebook is unresolved: separation vs none,
36 vs 12 swar bins, lam 0 vs 1, augmentation vs none, c4h vs the bar. Some of those were
written up as findings before the noise floor was known. The val column, sd 0.011–0.030, and
`cv5` (pooled over 1810 clips) are the instruments for anything finer.

#### The instrument to use from here

`cv5` gave 0.443 for the c4h configuration, pooling out-of-fold predictions over all 1810
train clips — roughly four times the evaluation data of a single val split and no
peak-picking per fold. A `cv5` run of `aug_jitter` (~2 h) would settle the augmentation
question far better than more seeds on 150 test clips, and the same applies to the lam sweep
and the bin-count sweep. **Cheap and decisive beats cheap and noisy.**

For the record, the survey's defensible conclusions, all of them multiples of the noise:

1. The tonic must be in the representation — 2.0–2.7×, with shuffled controls at chance.
2. Conditioning on the tonic through FiLM is worse than not using it at all.
3. A template-scoring head beats a linear classifier by +0.115; the templates may be learned.
4. Source separation does not help; speech-pretrained distilHuBERT does not transfer.
5. Everything finer than that awaits a better instrument than 150 clips.

### 2026-09-01 — Batch 6 [hybrid] launched, and a package-name collision worth knowing about

Stage 5 is the brief's stretch goal — *"only if nothing here beats motif-classifier"* — and
Batch 5 confirmed the condition: 0.373 ± 0.031 against a reported 0.400, indistinguishable.

`scripts/20_fuse_symbolic.py` fuses without training anything. It reads a finished run's
`best.pt`, refits the symbolic method on **the same 1350-clip fit half** (same seed, same
video-grouped split), scores the same 460 val and 150 test clips with both, converts each
score matrix to probabilities at a temperature fitted on val, then sweeps the mixing weight
on val and applies it once to test.

Two design points that matter for whether the number means anything:

* **Refit, do not reuse.** motif-classifier's published M14 was fitted on all 1810 train
  clips, which includes our 460 val clips. Fusing that against the DL model and choosing a
  weight on val would be choosing on data the symbolic half had memorised. It is refitted on
  the fit half instead.
* **Calibrate before adding.** The symbolic scores are unnormalised affinities and the DL
  logits are not calibrated either. Summing them raw would hand the fusion to whichever has
  the larger numeric scale, and the "best weight" would be measuring scale, not skill.

**The collision.** The first attempt died on `ModuleNotFoundError: models.gamadhani`. Both
projects define a top-level package called `models` — ours at `survey-aug-2026/models/`, and
the one `represent.py` reaches via `pipeline.estimate_tonic_hz` at
`melody-first/sequence/models/`. Whichever is imported first wins for the whole process, so
importing our trainer and *then* motif-classifier makes Python look for `gamadhani` inside
our package, where it correctly does not exist. Nothing was broken; the two halves simply
cannot share a `sys.modules` entry.

Fixed with a `_shadowed("models")` context manager that evicts the cached package while the
symbolic side imports and restores ours afterwards, so each half gets the package it means.
Renaming either package would have been cleaner and touches code this folder does not own.

This is the cost of the "import from the sibling projects rather than reimplementing" rule,
and it is still the right rule — but a second project with a `models/`, `utils/` or
`common/` will hit exactly this, and the fix is the shadowing helper, not a subprocess.

### 2026-09-01 — Batch 6 [hybrid]: fusion is the largest effect in the survey, and it replicates

`aug_jitter` fused with M14, at three seeds, weight swept on val and applied once to test.

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| DL alone, val | 0.467 | 0.415 | 0.417 | 0.433 | 0.030 |
| **fused, val** | 0.570 | 0.546 | 0.524 | **0.546** | 0.023 |
| DL alone, test | 0.400 | 0.340 | 0.380 | 0.373 | 0.031 |
| **fused, test** | 0.440 | 0.440 | 0.460 | **0.447** | 0.012 |

val **+0.113 ± 0.022** (t = 5.2) · test **+0.073 ± 0.019** (t = 3.9)

Both significant, unlike every fine distinction this survey has chased since Stage 2. The
fused test mean is **0.447, 95 % CI [0.418, 0.475]** — the interval clears the entire
symbolic family, whose test scores run 0.373 (M14) to 0.400 (m9plus, m12).

The chosen weight lands at 0.55-0.70 across seeds and the sweep is a plateau rather than a
spike, so the optimum is not an artifact of picking the best point on a jagged curve. A
fourth fusion with a different DL parent (`c4h`, no augmentation) gives val 0.552 / test
0.427 — the effect is a property of combining the two families, not of one checkpoint.

**Fused test variance is a third of the DL model's** (sd 0.012 against 0.031). Averaging two
partly independent error patterns stabilises the prediction as well as improving it, which
is what a genuine ensemble gain looks like and what a lucky seed does not.

#### What this says about the two families

Neither parent is close: 0.433 val for the CQT net, 0.502 for M14 scored on the same clips.
Together, 0.546. The gain is not a tiebreak between two views of the same evidence — they
are reading different things. The CQT net sees Sa-anchored spectral energy and no pitch
track; M14 sees CREPE note events and no spectral content. **The single most useful thing
this survey found is that these two are complementary**, and it was the brief's stretch goal,
attempted only after everything else had been exhausted.

It also reframes the bar. The comparison was never "DL versus symbolic" — it is that the DL
model contributes something the symbolic pipeline does not have, worth +0.073 test on top of
it, and vice versa.

#### The bar, corrected

motif-classifier's `final.json` records per-method test scores that were not being read
correctly earlier in this notebook. **M14, its best method on CV (0.468), scores 0.373 on
test** — not 0.400. The 0.400 figure belongs to m9plus and m12, neither of which is the
CV-best method.

So the symbolic side shows the same CV→test inversion measured on our side, and its three
strongest methods span 0.373-0.400 — one noise width. Every "the bar is 0.400" statement in
this notebook should be read as "the symbolic family lands in 0.373-0.400 on 150 clips".

| | test top-1 |
|---|---|
| symbolic family (m11, m12, m9, m9plus, m14) | 0.373 - 0.400 |
| DL alone, best config, 3 seeds | 0.373 ± 0.031 |
| **fused, 3 seeds** | **0.447 ± 0.012** |


### 2026-09-01 — Batch 7 [hybrid] launched: the histogram as an *input*, not a second opinion

Batch 6 answered "do the two families agree in different places?" — yes, worth +0.073 test.
It did not answer the brief's actual Stage 5 question, which is whether the naive melody
feature helps a network *while it trains*. The two are different experiments:

| | what it can express |
|---|---|
| fusion (Batch 6) | a weighted average of two finished 50-way opinions |
| concatenation (Batch 7) | the head reads both representations at once, so it can learn "this CQT pattern means Bageshree **only when** the histogram shows a weak Ga" |

Fusion is the safer bet — it cannot overfit, since it adds one parameter swept on val.
Concatenation has the higher ceiling and the obvious failure mode: 120 extra input
dimensions on 1350 clips is a way to memorise videos.

#### What was built

`--melody` on `10_train.py`, working for any architecture and either head. The vector is
`common/melody.py` — **the same 120-bin histogram the Stage 0 probe scored**, moved out of
the probe script so the network and the probe cannot drift apart. It reaches the head
through a small encoder (`heads.SideFeatures`, 120 → 64), because a normalised histogram
concatenated raw onto 432 learned activations is diluted to nothing by the first Linear.

Applied *after* FiLM: the tonic conditions what the backbone heard, while the histogram is
already expressed relative to Sa and has nothing left to condition.

#### The control, run first

A hybrid number is uninterpretable without both halves measured the same way. The CQT half
was already on the board; `scripts/21_melody_only.py` supplies the other — logistic
regression on the histogram, fitted on the same 1350 clips, scored on the same val and test.

| | val top-1 | test top-1 |
|---|---|---|
| melody histogram alone, 3 seeds | 0.396 ± 0.032 | **0.373 ± 0.024** |
| CQT net alone (aug_jitter), 3 seeds | 0.433 ± 0.030 | **0.373 ± 0.031** |
| M14, the symbolic champion | — | 0.373 |

**The two branches are exactly equally strong on test, and a 120-bin histogram with no
musical knowledge in it matches the symbolic pipeline's best method.** Everything M14 does
after the histogram — note segmentation, n-grams with skips, the 12-way tonic search, the
chi-square DB templates — is worth nothing on the held-out 150 relative to the raw pitch
mass, once the tonic is a hand annotation rather than an estimate. That is a result in its
own right and it belongs on the record whatever Batch 7 does.

It also sharpens what Batch 6 measured: fusion of two 0.373 branches reaching 0.447 is a
combination effect, not one strong model dragging a weak one along.

#### Queued

`hybrid_feat` is `aug_jitter` + `--melody`, so the only difference from a run already on the
board is the melody vector. Three seeds, because this survey has been fooled by a single
seed three times. `hybrid_nodb` drops the DB-template head — with a real pitch histogram as
an input, the head that learned to *infer* one may have nothing left to do.

Prediction, recorded before the runs finish: concatenation beats either branch but not
fusion's 0.447, because 1350 clips is too few to learn the interaction that is the only
thing concatenation can express and averaging cannot.


### 2026-09-01 — Batch 7 [hybrid]: concatenation loses to averaging, and the prediction was half wrong

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| melody_only | melody histogram alone, logreg *(control)* | 0.430 | 0.347 | - | 0.410 (0.267) |
| melody_only_seed1 | melody histogram alone, seed 1 | 0.367 | 0.393 | - | 0.433 (0.263) |
| melody_only_seed2 | melody histogram alone, seed 2 | 0.391 | 0.380 | - | 0.412 (0.261) |
| hybrid_feat | aug_jitter + **melody histogram as an input** | 0.457 | 0.380 | +0.346 | 0.438 (0.266) |
| hybrid_seed1 | hybrid_feat at seed 1 | 0.452 | 0.373 | +0.341 | 0.416 (0.267) |
| hybrid_seed2 | hybrid_feat at seed 2 | 0.446 | 0.340 | +0.335 | 0.422 (0.265) |
| hybrid_nodb | hybrid_feat without the DB-template head | 0.441 | 0.327 | +0.330 | 0.403 (0.264) |

Three seeds each, against the two branches and against Batch 6's fusion:

| | val top-1 | test top-1 |
|---|---|---|
| melody histogram alone | 0.396 ± 0.032 | 0.373 ± 0.024 |
| CQT net alone (aug_jitter) | 0.433 ± 0.030 | 0.373 ± 0.031 |
| **hybrid — histogram as an input** | **0.451 ± 0.005** | **0.364 ± 0.021** |
| **fusion — the two averaged (Batch 6)** | **0.546 ± 0.023** | **0.447 ± 0.012** |

| | val | test |
|---|---|---|
| hybrid − CQT alone | +0.018 (t 1.1) | −0.009 (t −0.4) |
| hybrid − melody alone | +0.055 (t 3.0) | −0.009 (t −0.5) |
| **fusion − hybrid** | **+0.095 (t 7.0)** | **+0.082 (t 5.9)** |

**On test the hybrid is indistinguishable from either of its own halves, and fusion beats it
decisively.** The prediction above was half right — it does lose to fusion — and half wrong:
it does not beat either branch. Giving the network the histogram as an input recovers what
the histogram already knew and nothing more.

`hybrid_nodb` (0.441 val / 0.327 test, one seed) is at or below the DB-head version, which
is the third independent sign that the DB-template head is not doing the work its Stage 4
result suggested.

#### Why averaging beats concatenating, measured rather than asserted

The two branches disagree constantly: they pick the same raag on **28.7 %** of the 150 test
clips, and both are right on only 20 %. Either-one-right — the oracle a perfect selector
would reach — is **0.547**. So there is a large pool of complementary evidence, and the
question is how much of it each combination method harvests:

| | test top-1 | share of the 0.547 oracle it captures | clips right that *neither* branch got |
|---|---|---|---|
| hybrid (concatenated) | 0.380 | 0.537 | **13** |
| fusion (averaged) | 0.440 | **0.732** | 6 |

The hybrid finds *more* genuinely new answers than fusion does — 13 clips against 6, which is
the interaction effect concatenation is supposed to buy and averaging cannot express. It
just loses far more of what the branches already had. Its predictions agree with the melody
branch (0.427) more than with the CQT branch (0.307): with 120 clean, immediately usable
input dimensions available, the trunk has little gradient pressure to keep improving, and
the best checkpoints land at **epoch 3, 7 and 8** — against the teens and twenties for the
same configuration without the histogram. It converges onto the easy feature and stops.

This is a real result about the method and not a tuning failure, but it is the version of
the failure that is fixable: the fix is to stop the shortcut, not to add capacity. Two ways,
neither run — modality dropout (zero the histogram on a fraction of training steps, so the
trunk must stay useful alone), and training the trunk first and the joint head second.

#### Where the survey now stands

| | test top-1 |
|---|---|
| symbolic family (m11, m12, m9, m9plus, m14) | 0.373 – 0.400 |
| melody histogram alone, logreg, 3 seeds | 0.373 ± 0.024 |
| CQT net alone, best config, 3 seeds | 0.373 ± 0.031 |
| hybrid, histogram as an input, 3 seeds | 0.364 ± 0.021 |
| **fusion of the two, 3 seeds** | **0.447 ± 0.012** |
| oracle: either branch right | 0.547 |

Everything that is not a *combination* of the two families sits at 0.373 ± noise, including
a 120-bin histogram with no musical knowledge in it. Combining them is the only thing in
this survey that moved the number, and averaging their outputs — the cheapest method
available, one parameter swept on val, no training — remains the best way found to do it.


### 2026-09-01 — a correction, and a portable partner for the fusion

**`--gain-jitter` never did anything to a CQT run.** `make_dataset` passes `gain_jitter_db`
to `WaveformDataset` only; `CQTDataset` takes `freq_shift_bins` and nothing else. So
`aug_jitter` and every run descended from it is **pitch jitter alone** -- a roll of up to 2
CQT bins, 22 cents. The flag was accepted and ignored. Nothing that was concluded changes
(the augmentation was not significant either way), but "pitch/gain jitter" was wrong
wherever it appears above and should be read as "pitch jitter".

**A second fusion partner, for packaging.** M14 needs both pitch trackers -- pYIN through
the `vamp` native plugin for its notes, and torchcrepe for its histogram -- which makes it
awkward to hand to anyone else. `20_fuse_symbolic.py --symbolic melody` fuses the CQT net
with the naive histogram + logreg instead, which needs one pip package:

| fusion partner | val top-1, 3 seeds | test top-1, 3 seeds | seed 0 (the artifact) |
|---|---|---|---|
| M14 (motif-classifier) | 0.546 ± 0.023 | 0.447 ± 0.012 | 0.440 |
| melody histogram + logreg | 0.508 ± 0.035 | 0.440 ± 0.047 | 0.440 |

Identical at seed 0 and indistinguishable on the three-seed mean (test diff 0.007, se
0.028). M14 is a little steadier across re-deals -- its test sd is a quarter of the
histogram's -- which is worth knowing but is a statement about the method's variance, not
about the shipped model. The released artifact in `../best-model-09-01/` uses the histogram.

This is also the third piece of evidence that everything M14 does downstream of a pitch
histogram is not paying for itself on held-out data: the histogram matches it alone (0.373
each), matches it as a fusion partner, and beats it on dependencies.


### 2026-09-01 — the released model: `../best-model-09-01/`

The fusion, rebuilt as a standalone package (no imports outside its own directory) and
retrained by its own `train.py` on **all 1810 training clips** rather than the 1350-clip fit
half. Two runs, both from that script:

| | fit on | val top-1 | test top-1 | test top-5 |
|---|---|---|---|---|
| reproduction, holding out a fifth of the videos | 1350 | 0.522 | 0.447 | 0.793 |
| the survey's own equivalent (`fuse_aug_jitter_melody`) | 1350 | 0.548 | 0.440 | 0.793 |
| **released model** | **1810** | *(none held out)* | **0.487** | **0.820** |

**The standalone rewrite reproduces the survey.** 0.447 against 0.440 test, from code that
shares no files with it -- and the temperatures it fitted on val came out at 0.925 and 2.360,
the same values to three decimals. Remaining differences are batch order, a per-clip rather
than per-batch pitch roll, and a fixed 34 epochs instead of early stopping.

**Refitting on the val fifth as well is worth about +0.04 test** (0.447 -> 0.487), from 34 %
more training data. That is one noise width, so it is a soft claim; it is also the expected
direction and the standard recipe -- select the epoch count and the calibration constants on
a held-out fifth, then refit everything at those settings.

Two things the packaging exposed that the survey had not noticed:

* **Test clips are 53 s, not 20 s.** `clip_tensor` centre-crops to 20 s, so every model in
  this survey trained and was scored on the middle 20 s of each clip and ignored the rest.
  The released model's `predict` averages *all* the 20 s windows instead, which is the more
  natural thing to do with a user's recording. It scores 0.487 that way against 0.507 on the
  centre crop -- the same within noise, and the card reports 0.487 because that is what the
  shipped code does.
* **A train/eval mode bug of the kind that only bites at batch size 1.** A helper left the
  network in `train()` mode; the next single-window forward pass died inside batch norm. It
  was caught by checking that the raw-audio path and the cached-feature path agree, which is
  a check worth having wherever a model is repackaged.
* **torchcrepe dithers its output, and nothing in this repo knew.** `convert.bins_to_cents`
  decodes pitch onto a 20-cent grid and then adds triangular noise of +-20 cents to every
  frame to mask the quantisation, from numpy's unseeded global RNG. Our histogram bins are
  10 cents wide, so that noise is wider than two bins. Consequences, in order of how much
  they matter:

  1. **Inference was not reproducible.** The same clip gave top-1 probabilities of 0.43,
     0.61 and 0.46 in three processes, with the ranking below first place reshuffling.
     Seeded now in `melody_branch.f0_track`, with the caller's RNG state restored.
  2. **Every melody-histogram number in this notebook carries a dither draw** -- the Stage 0
     probe, `melody_only`, `hybrid_feat`, and the `--symbolic melody` fusions. It is one
     draw each, averaged over 1578 voiced frames per clip and 150-1810 clips, so it is
     noise on top of noise already accounted for, not a bias. But it is one more reason the
     fine distinctions this survey chased were never resolvable.
  3. Refitting the released model on seeded histograms moved its test score from 0.487 to
     0.480 -- one clip.

  The CQT branch retrained **bit-for-bit identically** across the two runs, which is the
  reassuring half of the story: the pipeline is deterministic everywhere we control it.


### 2026-09-10 — Batches 8–10 planned: Essentia, the full recordings, and a trunk nobody tuned

Five questions, each answered below either from what is already known or with the runs
that will answer it. The instrument, as established in Batches 4–5: a single seed resolves
about **0.06 on val**; three seeds resolve about **0.10 on test**. Everything below is read
against that.

| starting point (3 seeds) | val top-1 | test top-1 |
|---|---|---|
| CQT branch, aug_jitter | 0.433 ± 0.030 | 0.373 ± 0.031 |
| melody branch, CREPE histogram | 0.396 ± 0.032 | 0.373 ± 0.024 |
| fused (the released recipe) | 0.508 ± 0.035 | 0.440 ± 0.047 |

#### 1. Essentia instead of CREPE for the melody branch — Batch 8

On one Bageshree clip, Essentia's Melodia (Saraga's settings: hop 196, 10-cent bins, no
guessing through unvoiced frames) against CREPE-tiny:

| | speed | voiced | frame-to-frame steps over 100 c |
|---|---|---|---|
| Essentia Melodia | 57× real time | 70 % | **0.7 %** |
| CREPE tiny | 4× real time | 93 % | 5.3 % |

What should and should not matter: the melody branch reads an *octave-folded* histogram,
so the octave errors that dominate the contour plots cost it nothing. Melodia can only help
through **which frames get counted** — it leaves out a quarter of what CREPE's 0.4
confidence threshold admits, plausibly tanpura and accompaniment — and through the
non-octave spikes. (CREPE's ±20-cent dither and Melodia's 10-cent grid both blur by about
one bin, which the histogram's own smoothing absorbs.)

Paired runs, same three splits: `melody_only_essentia` against `melody_only`, and each
Essentia fusion against its CREPE twin. **Prediction:** the melody branch gains +0.02–0.04
val; the fusion gains less, because it already down-weights that branch's noise.
**Decision rule:** if Essentia wins on the paired comparison, it becomes the melody
branch's tracker, and the one for the full recordings too — at 57× real time, the 159 h
are about 3 CPU-hours; CREPE would need about 40.

Environment: essentia 2.1b6.dev1389 installed with pip into the poetry env. It is not in
`pyproject.toml`.

#### 2. The full recordings — Batch 10

All 412 Hub videos have their full recording on disk, filed under the same raag every
time. 22 further recordings belong to no split; they are listed in
`splits/v1.1_unassigned.csv` and nothing reads them. The partition is now written down in
`splits/v1.1_video_splits.csv` (fit / val / test per video, seeds 0–2), and it matches a
fresh derivation from the Hub clips.

| role | videos | full recordings | what the Hub shows the model |
|---|---|---|---|
| fit | 270 | 102.6 h | 7.5 h (7.3 %) |
| val | 92 | 37.9 h | 2.6 h (6.7 %) |
| test | 50 | 18.3 h | 0.8 h (4.5 %) |

**Design.** Same fit videos. `common/fullaudio.py` caches each recording's Sa-anchored CQT
once, then serves fresh random 20 s windows every epoch. Window features match the Hub
pipeline's to within one storage step (max difference 0.0037 on a checked window). A gentle
filter skips the first and last 30 s and any window less than 80 % loud (within 20 dB of the
recording's median frame); 85 % of positions pass on the first recording checked.

Validation and test remain the Hub clips, so every number stays comparable with every
earlier run. `92_score_full.py` adds a second view: 20 fixed windows of every val and test
recording, with the Hub-trained baselines scored the same way.

| run | what it isolates |
|---|---|
| `full_aug` ×3 seeds | the headline: 20 windows/video/epoch, 20 epochs |
| `full_w5` | aug_jitter's exact step count, different audio: diversity alone |
| `full_nofilter` | what the filter is worth |
| `full_wide` | whether 13.7× the audio supports a bigger trunk |

**What this can and cannot fix.** More windows per video is not more videos. The fit set is
still 270 performances — often one or two artists per raag — and everything so far has
pointed at a ceiling set by data. If that ceiling is *audio per performance*, `full_aug`
moves; if it is *number of performances*, it will not, and the only fix is more
recordings. That distinction is the answer to whether the dataset should be upgraded, so
it matters more than the headline number. **Prediction:** val +0.05–0.10 over aug_jitter,
most of it visible in `full_w5` too. Under +0.03 means the ceiling is performances.

#### 3. Melodic order in the CQT branch — Batch 9, pooling

**Not tried before.** The trunk is not order-blind: its temporal receptive field is 80 frames,
**3.7 s**, with a stride of 0.74 s, so each of the ~26 positions left in a 20 s window already
encodes a few seconds of melody in order. What the final mean discards is order *between*
those stretches. The temporal swar head proposed under Next steps §3 is the full version of
this idea, and it was never run. These are the cheap versions:

| pool | order-sensitive | receptive field | params |
|---|---|---|---|
| `mean` (current) | within 3.7 s | 3.7 s | 496 k |
| `tconv` — dilated convs 1, 2, 4, residual, zero-initialised | yes | ~14 s | 817 k |
| `gru` — bidirectional, concatenated with the mean | yes | 20 s | 927 k |
| `attn` *(control)* | no — weights moments | 3.7 s | 551 k |
| `stats` *(control)* | no — adds variance | 3.7 s | 496 k |

A layer whose kernel spans the entire time axis is left out on purpose. It learns a
separate weight for each absolute position, and a window cropped from a random point in a
performance has no meaningful absolute positions: the same phrase at second 3 and at
second 13 would become different features. `tconv` and `gru` test the underlying
hypothesis while staying shift-invariant.

**Prediction and ceiling.** Order should fix the same-scale confusions — Bhupali/Deshkar,
Bageshree/Bheempalasi — which are 10.9 % of test errors (8.4× chance). Fixing every one of
them moves test by +0.066. So even perfect order modelling is a modest gain on this split.
Predict ≤ +0.03 val. If `attn` or `stats` gain as much as `tconv` or `gru`, the gain was a
richer summary, not order.

#### 4. Where the CQT network came from — Batch 9, shape

**Assembled, not taken from a paper, and never swept.** The parts are standard: a 5×5 stem,
pre-activation residual blocks of 3×3 convolutions with batch norm, and the music-CNN habit
of pooling time harder than frequency, because here frequency is the label. The numbers are
mine: 4 blocks, channels 32/64/96/128, a 24-channel projection, dropout 0.1, frequency
halved in three blocks. Every sweep in this survey — `db_lam`, swar bins, jitter, seeds —
happened downstream of that trunk.

Working this answer out turned up the choice I would now question most. **Three frequency
halvings leave 18 cells for 4 octaves, about 2.7 semitones each** — coarser than the swar
grid the head is supposed to read, so sub-cell pitch has to be carried in the channels.

A one-factor sweep around the default, single seed:

| run | change | params |
|---|---|---|
| `arch_w05` / `arch_w2` | width ×0.5 / ×2 | 124 k / 1.98 M |
| `arch_d3` / `arch_d5` | 3 / 5 blocks | 224 k / 932 k |
| `arch_f36` / `arch_f72` | 36 / 72 frequency cells | 496 k — same convs; the head's input grows to 864 / 1728 |

**Decision rule** for Batch 9 as a whole: anything ≥ +0.03 val over aug_jitter's seed 0 gets
seeds 1 and 2; the winner then runs on the full recordings.

#### 5. Decisions revisited — and ones deliberately left alone

- **The gentle filter** is the one new data-quality choice the full recordings force: raw
  concert video has dead air and applause at its edges. On, with `full_nofilter` as its
  ablation. Speech and applause detection were not attempted — that needs a classifier and
  a threshold nobody has checked.
- **Epoch budget.** The full-audio epochs are 4× longer, so 20 epochs with patience 6.
  `full_w5` keeps the old schedule for a like-for-like comparison.
- **Capacity.** Width is the setting most likely to change with more data, hence
  `full_wide`.
- **HPSS: not re-run.** On clips it gave c3 0.304 vs c2 0.302 val, and r3 was worse.
  Re-testing a measured null on 159 h would cost ~2.5 CPU-hours and 1.8 GB on a disk with
  11 GB free.
- **Band-limiting is irrelevant to the CQT branch by construction.** Its top bin is 16× its
  `fmin`, at most **1.76 kHz**, so it never sees anything a 16 kHz resample removes. The CQT
  branch's val gain attributed to band-limiting in
  `../augmentation-utils/deidentifier.md` §9.3 (0.443 → 0.478) therefore cannot come from
  what was removed; it is a different random draw (~1.5 SE). The de-identification cost in
  §9.2 (0.045) can still be real, since whitening changes the harmonic balance inside the
  band.
- **De-identification as a training augmentation.** §9.4 finds it safe, and a light
  sprinkling (≈ a quarter of training windows) is a reasonable next step. It needs a
  de-identified CQT cache of the full recordings — another 1.8 GB — so it waits. 🟥
- **Essentia's settings** stay at Saraga's, so the tracker tested is the one whose contours
  raised the question.

**Order and cost.** GPU: Batch 10 (~6.5 h), then Batch 9 (~7 h). CPU, alongside: Batch 8
(~30 min). The full-recording CQT cache is 1.8 GB, and the disk now has about 9 GB free.


### 2026-09-10 — Batch 8: Essentia beats CREPE on validation, and test cannot tell them apart

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| melody_only | melody histogram alone, logreg *(control)* | 0.430 | 0.347 | - | 0.410 (0.267) |
| melody_only_seed1 | melody histogram alone, seed 1 | 0.367 | 0.393 | - | 0.433 (0.263) |
| melody_only_seed2 | melody histogram alone, seed 2 | 0.391 | 0.380 | - | 0.412 (0.261) |
| melody_only_essentia | Essentia histogram alone, logreg | 0.491 | 0.307 | - | 0.423 (0.268) |
| melody_only_essentia_seed1 | Essentia histogram alone, seed 1 | 0.409 | 0.347 | - | 0.432 (0.266) |
| melody_only_essentia_seed2 | Essentia histogram alone, seed 2 | 0.448 | 0.367 | - | 0.413 (0.263) |
| fuse_aug_jitter_melody | aug_jitter + CREPE histogram, fused | 0.548 | 0.440 | - | 0.430 (0.270) |
| fuse_aug_seed1_melody | aug_seed1 + CREPE histogram, fused | 0.485 | 0.393 | - | 0.453 (0.267) |
| fuse_aug_seed2_melody | aug_seed2 + CREPE histogram, fused | 0.491 | 0.487 | - | 0.430 (0.264) |
| fuse_aug_jitter_melody_essentia | aug_jitter + **Essentia** histogram, fused | 0.565 | 0.413 | - | 0.435 (0.270) |
| fuse_aug_seed1_melody_essentia | aug_seed1 + Essentia histogram, fused | 0.520 | 0.380 | - | 0.445 (0.270) |
| fuse_aug_seed2_melody_essentia | aug_seed2 + Essentia histogram, fused | 0.526 | 0.427 | - | 0.430 (0.265) |

| three seeds | val top-1 | test top-1 |
|---|---|---|
| melody branch, CREPE | 0.396 ± 0.032 | 0.373 ± 0.024 |
| **melody branch, Essentia** | **0.449 ± 0.041** | 0.340 ± 0.031 |
| fused, CREPE | 0.508 ± 0.035 | 0.440 ± 0.047 |
| **fused, Essentia** | **0.537 ± 0.024** | 0.407 ± 0.024 |

The means point in opposite directions on the two splits, so the paired comparison is the one
to read — same clips, same classifier, only the tracker differs:

| seed | val diff | only E right / only C right | exact p | test diff | only E / only C | exact p | test 95 % CI (video bootstrap) |
|---|---|---|---|---|---|---|---|
| 0 | **+0.061** | 68 / 40 | **0.009** | −0.040 | 15 / 21 | 0.41 | [−0.127, +0.053] |
| 1 | +0.041 | 69 / 50 | 0.10 | −0.047 | 16 / 23 | 0.34 | [−0.133, +0.047] |
| 2 | **+0.057** | 73 / 47 | **0.022** | −0.013 | 20 / 22 | 0.88 | [−0.113, +0.087] |

*(melody branch alone; E = Essentia, C = CREPE)*

**Validation is the comparison with power, and it is clean.** The melody branch fits
nothing on val — the logistic regression sees only the fit clips — so there is no selection
optimism in these val numbers. The three val sets are three different deals of 92 videos,
and all three favour Essentia. Pooled over seeds, Essentia alone is right on 210 clips and
CREPE alone on 137 (p ≈ 1e-4 if the three sets were independent; they overlap, so the true
evidence is somewhat weaker).

**Test does not contradict it; it cannot resolve it.** All three test scores come from the
same 50 videos, so they are one measurement taken with three models, not three
measurements. None of the three differences is significant. Every video-bootstrap interval
contains both zero and the size of the val effect.

**The mechanism is not coverage.** Across the corpus, Essentia counts 74 % of frames voiced
against CREPE's 80 % (medians), and neither tracker leaves any clip with under 3 s of voiced
pitch. The planning entry's "a quarter fewer frames" came from a single clip and overstated
the gap. So the gain lies in *which* frames are counted and how cleanly — fewer non-octave
spikes — not in how many.

**Prediction check.** Predicted: melody branch +0.02–0.04 val, the fusion less. Measured:
+0.053 and +0.029. Direction and ordering right; the size was underestimated.

**Decision, by the rule set before the runs:** Essentia becomes the development pipeline's
melody tracker, passed as `--tracker essentia`. The defaults stay CREPE, so every earlier
command still reproduces its result. The released model is not changed on the strength of
a validation-only result.

**Next, awaiting a go-ahead:** Essentia tracks of the full recordings — about an hour of CPU
and about 0.3 GB of disk — for a melody branch trained on the full recordings, and a
full-audio fusion to go with Batch 10.


### 2026-09-10 — how many clips per performance? The model already knows its performances

Asked after Batch 10's first seed: how many training clips per performance should a
dataset keep? Two things first.

**A correction about epochs.** A Hub epoch is the 1,350 fixed clips once each; a full-audio
epoch is 5,400 fresh windows. Counted in windows seen, `full_aug` and `aug_jitter` learn at
the *same* speed: 0.45 val after ~32k and ~30k presentations respectively. "Epoch 5
instead of 30", as I first put it, compared clocks of different sizes.

**The diagnostic that answers it.** Score each model on 5 evenly spaced windows of every
*training* performance — audio `aug_jitter` never heard, since the Hub kept only 100 s of
each recording — and on 5 windows of every *validation* performance. The first number is
how well a model generalises within a performance it has studied; the second, to
performances it has never met.

| model | unseen windows, **own** training performances | windows, **new** performances (val) |
|---|---|---|
| aug_jitter — the Hub's 5 clips (100 s) per performance | 0.782 per window · **0.944** per performance | 0.424 · 0.576 |
| full_aug — fresh windows from the whole recording | 0.835 · **0.967** | 0.450 · 0.565 |

*(one seed each; 1,350 windows for training performances, 460 for val.
`full_aug`'s "unseen" windows had usually been seen in training, so for it that column is
an upper bound.)*

**From 100 s per performance, the model already names the raag of 94 % of its training
performances from audio it never heard.** Whatever a performance can teach about its raag,
five clips have mostly taught it. Thirteen times the audio raised that to 97 %, and moved
new performances by +0.026 per window and −0.011 per performance — noise. **The gap that
matters, 0.78 against 0.42, is between performances, and more audio per performance does
not close it.** This is the ceiling everything since Batch 4 has pointed at, now measured
directly: the fit set is 270 performances, and that number is what limits the model.

**Recommendation for the dataset.** Keep about 5–10 clips per performance, and spend any
upgrade on *more performances* — more recordings per raag, and especially more artists per
raag, since one or two artists per raag also lets the model learn voices instead of raags.
The 22 unassigned recordings (`splits/v1.1_unassigned.csv`) are a free first step. Batch 11
(N = 5, 10, 20, 40 fixed windows per performance, equal budget) will turn "about 5–10" into
a measured curve. **Prediction:** flat within noise from N = 10 on.

**A correction to the release entry.** It says "test clips are 53 s, not 20 s", and that
every model "trained and was scored on the middle 20 s of each clip and ignored the rest".
Measured over all clips, that is wrong: training and test clips alike have a **median of
20.0 s**, and only 16 % (train) and 18 % (test) are longer. The 53 s figure came from the
first few clips opened. So the centre crop discarded a little audio from a minority of
clips in both splits, evenly. It is not a train/test asymmetry, and it does not change the
released model's reported score, which was measured through `predict` in any case.


### 2026-09-11 — Batch 10, three seeds: the full recordings give a small, consistent gain, and it survives fusion

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| full_aug | aug_jitter on **full recordings**, 20 windows/video/epoch | 0.480 | 0.433 | +0.370 | 0.451 (0.268) |
| full_aug_seed1 | full_aug at seed 1 | 0.474 | 0.433 | +0.363 | 0.427 (0.264) |
| full_aug_seed2 | full_aug at seed 2 | 0.439 | 0.433 | +0.328 | 0.417 (0.264) |
| fuse_full_aug_melody | full_aug + CREPE histogram, fused | 0.550 | 0.487 | - | 0.443 (0.270) |
| fuse_full_aug_seed1_melody | full_aug_seed1 + CREPE histogram, fused | 0.528 | 0.507 | - | 0.442 (0.267) |
| fuse_full_aug_seed2_melody | full_aug_seed2 + CREPE histogram, fused | 0.511 | 0.460 | - | 0.428 (0.265) |
| fuse_full_aug_melody_essentia | full_aug + **Essentia** histogram, fused | 0.563 | 0.447 | - | 0.431 (0.271) |
| fuse_full_aug_seed1_melody_essentia | full_aug_seed1 + Essentia histogram, fused | 0.546 | 0.467 | - | 0.438 (0.268) |
| fuse_full_aug_seed2_melody_essentia | full_aug_seed2 + Essentia histogram, fused | 0.539 | 0.493 | - | 0.434 (0.265) |

Paired by seed, so each full-trained model faces the Hub-trained model on exactly its split:

| three seeds | trained on Hub clips | trained on full recordings | paired diff |
|---|---|---|---|
| CQT branch alone, val | 0.433 ± 0.030 | 0.464 ± 0.022 | **+0.031** (t 2.2) |
| CQT branch alone, test | 0.373 ± 0.031 | 0.433 ± 0.000 | **+0.060** (t 3.4) |
| fused with CREPE, val | 0.508 ± 0.035 | 0.530 ± 0.020 | +0.022 (t 1.8) |
| fused with CREPE, test | 0.440 ± 0.047 | 0.484 ± 0.023 | +0.044 (t 1.1) |
| fused with Essentia, val | 0.537 ± 0.024 | **0.549 ± 0.012** | +0.012 (t 1.5) |
| fused with Essentia, test | 0.407 ± 0.024 | 0.469 ± 0.023 | +0.062 (t 4.0) |

*(t has 2 degrees of freedom: t 4.0 is p ≈ 0.06. The three test numbers in any row come
from the same 50 videos.)*

**11 of 12 paired comparisons are positive, and none is resolvable on its own.** The
direction is not in doubt; the size is a few points, below what three seeds on this corpus
can pin down.

**Prediction check.** Predicted +0.05–0.10 val for the CQT branch, with "under +0.03 means
the ceiling is performances". It landed at +0.031 — on the line. Read together with the
within-/across-performance diagnostic (0.78 against 0.42), the picture is consistent: audio
per performance is worth a few points, and the number of performances sets the ceiling.

**Fusion absorbs part of the val gain but not the test gain.** Alone, the CQT branch gains
+0.031 val; fused, +0.012–0.022. Part of what the extra audio teaches the CQT net, the
melody branch was already supplying. On test the gain stays at about +0.05 either way.

**The models are steadier.** Test sd for the CQT branch fell from 0.031 to 0.000 — all three
seeds scored exactly 65 of 150. It is the same variance reduction fusion showed, from a
different source: more data gives less arbitrary decision boundaries.

**Essentia against CREPE, again split.** Fused with the full-trained CQT branch, Essentia
wins on val (0.549 vs 0.530) and CREPE wins on test (0.484 vs 0.469). This is the same
pattern as Batch 8, and the same reading applies: val is the powered comparison, and test
cannot resolve a gap this small.

**Where development stands.** Chosen on val, the best configuration is **full-recording
CQT branch + Essentia melody branch, fused: val 0.549 ± 0.012, test 0.469 ± 0.023**. That
is not comparable with the released model (test 0.480), which was refit on all 362
train-pool videos; the like-for-like candidate would be the same refit with full recordings
and Essentia. That is a release decision, and nothing here forces it yet.

**Still running in Batch 10:** `full_w5` (5 fresh windows per epoch on aug_jitter's schedule —
does the gain come from more distinct audio or from more steps?), `full_nofilter`, and
`full_wide`. Batch 11 follows, then Batch 9.


### 2026-09-11 — Batch 10 complete: the controls, and what whole-recording evaluation shows

The same runs scored two ways: on the Hub clips as always, and on 20 fixed windows of every
val and test recording (`92_score_full.py`), where "per performance" is the vote over
those 20.

| run | what | val (Hub clips) | test (Hub clips) | val (20 win/video) | val video vote | test (20 win/video) | test video vote |
|---|---|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | 0.436 | 0.554 | 0.349 | 0.480 |
| aug_seed1 | aug_jitter at seed 1 | 0.415 | 0.340 | 0.408 | 0.489 | 0.325 | 0.400 |
| aug_seed2 | aug_jitter at seed 2 | 0.417 | 0.380 | 0.411 | 0.543 | 0.346 | 0.460 |
| full_aug | aug_jitter on **full recordings**, 20 windows/video/epoch | 0.480 | 0.433 | 0.463 | 0.598 | 0.404 | 0.540 |
| full_aug_seed1 | full_aug at seed 1 | 0.474 | 0.433 | 0.479 | 0.630 | 0.416 | 0.520 |
| full_aug_seed2 | full_aug at seed 2 | 0.439 | 0.433 | 0.423 | 0.543 | 0.397 | 0.520 |
| full_w5 | full recordings, 5 windows/video/epoch (aug_jitter's step count) | 0.474 | 0.413 | 0.449 | 0.554 | 0.364 | 0.420 |
| full_nofilter | full_aug without the trim/loudness filter *(ablation)* | 0.496 | 0.447 | 0.466 | 0.609 | 0.417 | 0.600 |
| full_wide | full_aug at double width | 0.493 | 0.447 | 0.475 | 0.641 | 0.395 | 0.520 |

**On whole recordings the full-trained models are clearly better.** Three seeds, paired:

| | Hub-trained | full-trained | paired diff |
|---|---|---|---|
| val, per window | 0.418 ± 0.015 | 0.455 ± 0.029 | +0.037 (t 2.0) |
| val, per performance | 0.529 ± 0.035 | 0.591 ± 0.044 | +0.062 (t 1.5) |
| test, per window | 0.340 ± 0.013 | 0.406 ± 0.010 | **+0.066** (t 5.2) |
| test, per performance | 0.447 ± 0.042 | **0.527 ± 0.012** | **+0.080** (t 4.0) |

**Read this with its bias in view.** The windows are drawn from the whole recording — the
same distribution the full-trained models learned from — while the Hub-trained models only
ever saw clips from wherever the Hub cut them. So part of this gain is the models matching
the evaluation, not raag knowledge the Hub clips could not teach. It is also a real,
practical gain: someone humming into `quickstart.py` sings from anywhere in a performance.
**For a deployed model, train on the full recordings.** For the question of what limits
raag recognition, the Hub-clip columns stay the neutral measure, and they moved by +0.031
val.

**The three controls** (one seed each; read against `full_aug` seed 0 — Hub val 0.480, test
0.433):

| run | Hub val | Hub test | full-window test (per perf.) | reading |
|---|---|---|---|---|
| `full_w5` — 5 fresh windows/epoch on aug_jitter's schedule | 0.474 | 0.413 | 0.364 (0.420) | at the Hub's step count, fresh audio ≈ the Hub's fixed clips |
| `full_nofilter` — no trim, no loudness filter | 0.496 | 0.447 | 0.417 (0.600) | the filter does not help; nominally it hurts |
| `full_wide` — double width | 0.493 | 0.447 | 0.395 (0.520) | 13× the audio does not call for a bigger trunk |

- **`full_w5` is the most informative control.** With the same number of steps as the Hub
  run, drawing fresh windows instead of reusing 5 fixed clips changed val by +0.007. Most
  of what `full_aug` gained came with its larger per-epoch budget, not with the variety of
  the audio. That is one more piece of evidence that a performance's raag is learned from
  little of it.
- **The gentle filter goes.** It was added on reasoning (concert edges are tuning and
  applause), and removing it is nominally better on every column. The defaults stay as
  they are, so `full_aug` reproduces; new full-audio runs should pass
  `--trim-seconds 0 --loud-fraction 0`.
- **Width stays.** A performance-limited model has nothing to spend extra capacity on.
  Batch 9 will say whether the trunk's *shape* matters on the Hub clips.

Batch 11 (clips per performance) is running; Batch 9 follows.


### 2026-09-11 — Batch 11: ten clips per performance, then flat

The dataset-design question, asked directly. Each run trains on a **fixed** set of N evenly
spaced 20 s windows per performance, reused every epoch — what a dataset with N clips per
video would give — at a matched budget of ~65k windows seen. Val and test are the Hub
clips, plus the 20-window whole-recording scores.

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | +0.357 | 0.424 (0.266) |
| clips5 | fixed 5 windows/performance, spread over the recording | 0.435 | 0.407 | +0.324 | 0.432 (0.268) |
| clips10 | fixed 10 windows/performance | 0.504 | 0.433 | +0.393 | 0.432 (0.266) |
| clips20 | fixed 20 windows/performance | 0.493 | 0.387 | +0.383 | 0.440 (0.265) |
| clips40 | fixed 40 windows/performance | 0.485 | 0.427 | +0.374 | 0.452 (0.267) |
| full_aug | aug_jitter on **full recordings**, 20 windows/video/epoch | 0.480 | 0.433 | +0.370 | 0.451 (0.268) |

| run | what | val (Hub clips) | test (Hub clips) | val (20 win/video) | val video vote | test (20 win/video) | test video vote |
|---|---|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | 0.436 | 0.554 | 0.349 | 0.480 |
| clips5 | fixed 5 windows/performance, spread over the recording | 0.435 | 0.407 | 0.434 | 0.554 | 0.360 | 0.500 |
| clips10 | fixed 10 windows/performance | 0.504 | 0.433 | 0.487 | 0.630 | 0.402 | 0.560 |
| clips20 | fixed 20 windows/performance | 0.493 | 0.387 | 0.467 | 0.609 | 0.387 | 0.500 |
| clips40 | fixed 40 windows/performance | 0.485 | 0.427 | 0.482 | 0.620 | 0.397 | 0.500 |
| full_aug | aug_jitter on **full recordings**, 20 windows/video/epoch | 0.480 | 0.433 | 0.463 | 0.598 | 0.404 | 0.540 |

| clips per performance (seed 0) | Hub val | full-window val | full-window val, per performance | windows seen at best epoch |
|---|---|---|---|---|
| 5 — the Hub's own (aug_jitter) | 0.467 | 0.436 | 0.554 | 45.9k |
| 5 — evenly spaced (clips5) | 0.435 | 0.434 | 0.554 | 44.6k |
| **10** | **0.504** | **0.487** | **0.630** | 48.6k |
| 20 | 0.493 | 0.467 | 0.609 | 37.8k |
| 40 | 0.485 | 0.482 | 0.620 | 54.0k |
| fresh every epoch (full_aug) | 0.480 | 0.463 | 0.598 | 59.4k |

**One step, then a plateau.** Both N = 5 points sit at about 0.45 Hub val and 0.435
full-window val. Every point from N = 10 up sits at about 0.49 and 0.475 — a step of about
+0.04 on both measures. From 10 on, the four points lie within 0.024 of each other, which
is noise. The prediction, "flat within noise from N = 10 on", held.

**Which 5 clips does not matter.** The Hub's own 5 and 5 evenly spaced windows score the
same on whole recordings (0.436 against 0.434). Nothing about how the Hub chose its chunks
is special.

**Caveats.** One seed per point. The step rests on 2 points against 4, measured two ways
that agree, and at about the size one seed can resolve. `clips10` at seeds 1 and 2 is
queued after Batch 9, to be paired with `aug_seed1`/`aug_seed2` and
`full_aug_seed1`/`full_aug_seed2`. Test runs 0.387–0.433 with no trend: it cannot resolve
a 0.04 step.

**Recommendation, now measured: 10 clips per performance — 200 s, twice what the Hub keeps.**
Beyond that, more of the same performance buys nothing on this corpus; the plateau is the
performances ceiling measured from the other side. The recordings are already on disk, so
doubling clips per performance costs nothing to collect. Every further gain has to come
from new performances. For a deployed model, training on the full recordings is as good as
10 fixed clips and simpler, so the Batch 10 advice stands.


### 2026-09-11 — Batch 9: melodic order adds nothing, a richer summary might, and the trunk was already in the right place

Every run is aug_jitter with one change, at seed 0, on the Hub clips. The reference is
aug_jitter's own seed 0: val 0.467, test 0.400. Resolution is about 0.06 val.

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | +0.357 | 0.424 (0.266) |
| pool_tconv | aug_jitter + **dilated temporal convs** before the mean | 0.467 | 0.387 | +0.357 | 0.433 (0.268) |
| pool_gru | aug_jitter + **BiGRU over time** | 0.461 | 0.353 | +0.350 | 0.427 (0.264) |
| pool_attn | aug_jitter, attention over time *(control)* | 0.470 | 0.373 | +0.359 | 0.456 (0.265) |
| pool_stats | aug_jitter, time pooled as mean + std *(control)* | 0.507 | 0.440 | +0.396 | 0.434 (0.266) |
| arch_w05 | aug_jitter at half width | 0.446 | 0.393 | +0.335 | 0.431 (0.267) |
| arch_w2 | aug_jitter at double width | 0.457 | 0.373 | +0.346 | 0.445 (0.262) |
| arch_d3 | aug_jitter with 3 blocks | 0.452 | 0.393 | +0.341 | 0.443 (0.266) |
| arch_d5 | aug_jitter with 5 blocks | 0.433 | 0.333 | +0.322 | 0.430 (0.263) |
| arch_f36 | aug_jitter, 36 frequency cells (default 18) | 0.480 | 0.400 | +0.370 | 0.433 (0.265) |
| arch_f72 | aug_jitter, 72 frequency cells (~67 cents each) | 0.467 | 0.380 | +0.357 | 0.433 (0.266) |

| run | what | val (Hub clips) | test (Hub clips) | val (20 win/video) | val video vote | test (20 win/video) | test video vote |
|---|---|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | 0.436 | 0.554 | 0.349 | 0.480 |
| pool_tconv | aug_jitter + **dilated temporal convs** before the mean | 0.467 | 0.387 | 0.422 | 0.554 | 0.331 | 0.460 |
| pool_gru | aug_jitter + **BiGRU over time** | 0.461 | 0.353 | 0.418 | 0.630 | 0.342 | 0.520 |
| pool_attn | aug_jitter, attention over time *(control)* | 0.470 | 0.373 | 0.432 | 0.576 | 0.377 | 0.500 |
| pool_stats | aug_jitter, time pooled as mean + std *(control)* | 0.507 | 0.440 | 0.459 | 0.609 | 0.383 | 0.500 |
| arch_w05 | aug_jitter at half width | 0.446 | 0.393 | 0.407 | 0.500 | 0.343 | 0.460 |
| arch_w2 | aug_jitter at double width | 0.457 | 0.373 | 0.426 | 0.554 | 0.327 | 0.460 |
| arch_d3 | aug_jitter with 3 blocks | 0.452 | 0.393 | 0.433 | 0.554 | 0.358 | 0.560 |
| arch_d5 | aug_jitter with 5 blocks | 0.433 | 0.333 | 0.415 | 0.598 | 0.325 | 0.460 |
| arch_f36 | aug_jitter, 36 frequency cells (default 18) | 0.480 | 0.400 | 0.454 | 0.587 | 0.328 | 0.420 |
| arch_f72 | aug_jitter, 72 frequency cells (~67 cents each) | 0.467 | 0.380 | 0.444 | 0.554 | 0.388 | 0.540 |

| change | val | vs aug_jitter | test |
|---|---|---|---|
| `tconv` — order, over ~14 s | 0.467 | +0.000 | 0.387 |
| `gru` — order, over the whole window | 0.461 | −0.006 | 0.353 |
| `attn` — which moments count *(control)* | 0.470 | +0.003 | 0.373 |
| **`stats` — mean + std *(control)*** | **0.507** | **+0.040** | **0.440** |
| width ×0.5 | 0.446 | −0.021 | 0.393 |
| width ×2 | 0.457 | −0.010 | 0.373 |
| 3 blocks | 0.452 | −0.015 | 0.393 |
| 5 blocks | 0.433 | −0.034 | 0.333 |
| 36 frequency cells | 0.480 | +0.013 | 0.400 |
| 72 frequency cells | 0.467 | +0.000 | 0.380 |

**Melodic order beyond 3.7 s is not what this model is missing.** Both order-sensitive
summaries — dilated temporal convolutions that see ~14 s, and a BiGRU over the whole
window — gain nothing: +0.000 and −0.006. The prediction was ≤ +0.03, against a ceiling of
+0.066 test if every same-scale confusion were fixed. The trunk's own 3.7 s of ordered
context is apparently all the order a 20 s window can teach from 270 performances. This
does not rule out the full temporal swar head (Next steps §3) — a different mechanism,
supervised per frame — but it removes the cheap version of the argument for it.

**The one change past the bar is a control.** Mean + std over time gains +0.040 val and
+0.040 test. What the standard deviation adds is how much each (channel, frequency) feature
*varies* across the window: a held nyas against an oscillating andolan, slow alap against
fast taan. That is texture, not order. The planning entry anticipated exactly this reading:
"if stats or attn gain as much as tconv or gru, the gain was a richer summary, not order".
Here stats gains and the order models do not. It is still one seed, at about the size one
seed resolves, so by the rule set beforehand it gets seeds 1 and 2 before anything builds
on it.

**The trunk needed no retuning.** Width, depth and frequency resolution all land between
−0.034 and +0.013 of the default, none past noise; 5 blocks is nominally the worst. The
18-cell concern raised in planning did not bite — 36 or 72 frequency cells change nothing,
so the channels carry the sub-cell pitch. The honest summary of where the architecture came
from is now: chosen by reasoning, and a one-factor sweep afterwards found nothing better
nearby.

**Together with Batches 10 and 11:** capacity does not help, longer-range order does not
help, and more audio per performance helps a little. Every lever that stays inside the same
270 performances is small. **The number of performances is the ceiling.**

Queued: `pool_stats` at seeds 1 and 2, after the `clips10` replication. If it holds, stats
pooling on the full recordings is the obvious combination.


### 2026-09-11 — Both replications hold: 10 clips per performance, and mean + std pooling

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | +0.357 | 0.424 (0.266) |
| aug_seed1 | aug_jitter at seed 1 | 0.415 | 0.340 | +0.304 | 0.429 (0.265) |
| aug_seed2 | aug_jitter at seed 2 | 0.417 | 0.380 | +0.307 | 0.419 (0.263) |
| clips10 | fixed 10 windows/performance | 0.504 | 0.433 | +0.393 | 0.432 (0.266) |
| clips10_seed1 | fixed 10 windows/performance, seed 1 | 0.450 | 0.440 | +0.339 | 0.439 (0.267) |
| clips10_seed2 | fixed 10 windows/performance, seed 2 | 0.461 | 0.413 | +0.350 | 0.432 (0.263) |
| pool_stats | aug_jitter, time pooled as mean + std *(control)* | 0.507 | 0.440 | +0.396 | 0.434 (0.266) |
| pool_stats_seed1 | pool_stats at seed 1 | 0.489 | 0.360 | +0.378 | 0.454 (0.265) |
| pool_stats_seed2 | pool_stats at seed 2 | 0.435 | 0.413 | +0.324 | 0.429 (0.263) |

| run | what | val (Hub clips) | test (Hub clips) | val (20 win/video) | val video vote | test (20 win/video) | test video vote |
|---|---|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | 0.436 | 0.554 | 0.349 | 0.480 |
| aug_seed1 | aug_jitter at seed 1 | 0.415 | 0.340 | 0.408 | 0.489 | 0.325 | 0.400 |
| aug_seed2 | aug_jitter at seed 2 | 0.417 | 0.380 | 0.411 | 0.543 | 0.346 | 0.460 |
| clips10 | fixed 10 windows/performance | 0.504 | 0.433 | 0.487 | 0.630 | 0.402 | 0.560 |
| clips10_seed1 | fixed 10 windows/performance, seed 1 | 0.450 | 0.440 | 0.443 | 0.587 | 0.385 | 0.480 |
| clips10_seed2 | fixed 10 windows/performance, seed 2 | 0.461 | 0.413 | 0.424 | 0.533 | 0.416 | 0.580 |
| pool_stats | aug_jitter, time pooled as mean + std *(control)* | 0.507 | 0.440 | 0.459 | 0.609 | 0.383 | 0.500 |
| pool_stats_seed1 | pool_stats at seed 1 | 0.489 | 0.360 | 0.453 | 0.587 | 0.357 | 0.520 |
| pool_stats_seed2 | pool_stats at seed 2 | 0.435 | 0.413 | 0.407 | 0.576 | 0.383 | 0.500 |

Paired by seed (same split, same seed):

| three seeds | val | test | whole-recording test, per window | per performance |
|---|---|---|---|---|
| **10 clips vs the Hub's 5** | **+0.038** (t 14.7) | +0.056 (t 2.5) | +0.061 (t 12.4) | **+0.093** (t 7.0) |
| 10 clips vs the full recordings | +0.007 (t 0.5) | −0.004 | −0.005 | +0.013 |
| **mean + std vs mean pooling** | **+0.043** (t 2.6) | +0.031 (t 5.3) | +0.034 (t 23.6) | +0.060 (t 2.0) |

**The clips-per-performance step is the steadiest result in the survey.** From 5 clips per
performance to 10, val rises by +0.037, +0.035 and +0.043 in the three seeds, and the gain
shows up on every other measure too. Ten fixed clips then match the full recordings to
within noise on everything. The curve from Batch 11 stands, replicated: **10 clips per
performance, and flat beyond.**

**Mean + std pooling holds, and it is modest:** val +0.043, test +0.031, positive in 14 of
15 paired comparisons. It was the order-free control in Batch 9, so what it establishes is
that the *spread* of each feature over the window carries raag information the mean
throws away. It does not establish order.

#### The five questions of 2026-09-10, answered

| question | answer | evidence |
|---|---|---|
| 1. Essentia instead of CREPE? | Yes, for development: melody branch +0.053 val, in every seed. Test (50 videos) cannot resolve it. | Batch 8 |
| 2. How much can the full recordings give? | A few points, and all of it by **10 clips per performance** (+0.038 val, +0.093 per-performance test on whole recordings). Beyond that, flat. The ceiling is the **number of performances**: a Hub-trained model already names 94 % of its own training performances from audio it never heard, against 58 % of new ones. | Batches 10, 11; the within/across diagnostic |
| 3. Melodic order in the CQT pooling? | Not tried before; now tried. Order beyond the trunk's 3.7 s adds nothing (tconv +0.000, gru −0.006). An order-free richer summary, mean + std, adds +0.043 val, replicated. | Batch 9 |
| 4. Where did the CQT net come from? | Assembled by reasoning from standard parts, never swept. A one-factor sweep afterwards — width, depth, frequency cells — finds nothing better nearby. | Batch 9 |
| 5. Other decisions? | The gentle filter is unnecessary. Extra width does not help even with 13× the audio. HPSS was not re-run (a measured null). Band-limiting cannot affect the CQT branch, whose top bin is ≤ 1.76 kHz. | Batches 9, 10 |

**For the dataset:** keep 10 clips per performance — the recordings are already on disk, so
that costs nothing — and spend any further effort on **new performances**, above all new
artists per raag. The 22 unassigned recordings are the first ones available.

**Queued as Batch 12:** the two replicated wins together — 10 clips per performance with
mean + std pooling, three seeds — each fused with both melody branches, along with fusions
of plain `clips10` for the paired comparison. Both wins are order-free and work through
different mechanisms (more views of each performance; texture in the summary), so they
may add.


### 2026-09-11 — Batch 12: the two wins together, and the best configuration the survey has found

| run | what | val top-1 | test top-1 | vs stage 1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| clips10 | fixed 10 windows/performance | 0.504 | 0.433 | +0.393 | 0.432 (0.266) |
| clips10_seed1 | fixed 10 windows/performance, seed 1 | 0.450 | 0.440 | +0.339 | 0.439 (0.267) |
| clips10_seed2 | fixed 10 windows/performance, seed 2 | 0.461 | 0.413 | +0.350 | 0.432 (0.263) |
| clips10_stats | 10 clips/performance + **mean/std pooling** | 0.500 | 0.447 | +0.389 | 0.444 (0.267) |
| clips10_stats_seed1 | clips10_stats at seed 1 | 0.491 | 0.460 | +0.380 | 0.441 (0.268) |
| clips10_stats_seed2 | clips10_stats at seed 2 | 0.489 | 0.500 | +0.378 | 0.468 (0.264) |
| fuse_clips10_stats_melody | clips10_stats + CREPE histogram, fused | 0.537 | 0.487 | - | 0.452 (0.269) |
| fuse_clips10_stats_seed1_melody | clips10_stats_seed1 + CREPE histogram, fused | 0.533 | 0.460 | - | 0.448 (0.270) |
| fuse_clips10_stats_seed2_melody | clips10_stats_seed2 + CREPE histogram, fused | 0.539 | 0.520 | - | 0.468 (0.265) |
| fuse_clips10_stats_melody_essentia | clips10_stats + Essentia histogram, fused | 0.572 | 0.480 | - | 0.455 (0.270) |
| fuse_clips10_stats_seed1_melody_essentia | clips10_stats_seed1 + Essentia histogram, fused | 0.548 | 0.453 | - | 0.449 (0.272) |
| fuse_clips10_stats_seed2_melody_essentia | clips10_stats_seed2 + Essentia histogram, fused | 0.563 | 0.500 | - | 0.464 (0.266) |

| run | what | val (Hub clips) | test (Hub clips) | val (20 win/video) | val video vote | test (20 win/video) | test video vote |
|---|---|---|---|---|---|---|---|
| aug_jitter | c4h + pitch jitter | 0.467 | 0.400 | 0.436 | 0.554 | 0.349 | 0.480 |
| clips10 | fixed 10 windows/performance | 0.504 | 0.433 | 0.487 | 0.630 | 0.402 | 0.560 |
| pool_stats | aug_jitter, time pooled as mean + std *(control)* | 0.507 | 0.440 | 0.459 | 0.609 | 0.383 | 0.500 |
| clips10_stats | 10 clips/performance + **mean/std pooling** | 0.500 | 0.447 | 0.464 | 0.630 | 0.415 | 0.540 |
| clips10_stats_seed1 | clips10_stats at seed 1 | 0.491 | 0.460 | 0.486 | 0.641 | 0.414 | 0.620 |
| clips10_stats_seed2 | clips10_stats at seed 2 | 0.489 | 0.500 | 0.486 | 0.598 | 0.480 | 0.620 |

**Do they add?** Paired by seed, three seeds:

| clips10_stats against | val | test | whole-recording test | per performance |
|---|---|---|---|---|
| `clips10` (so: what mean + std adds) | +0.022 (t 1.6) | +0.040 (t 1.7) | +0.035 (t 2.3) | +0.053 |
| `pool_stats` (so: what the data adds) | +0.017 (t 0.9) | +0.064 (t 2.2) | +0.062 (t 3.3) | +0.087 (t 3.6) |
| **`aug_jitter`, the Hub baseline** | **+0.060** (t 4.4) | **+0.096** (t 3.9) | **+0.096** (t 4.8) | **+0.147** (t 3.1) |

On val the two are **sub-additive**: +0.038 and +0.043 alone, +0.060 together rather than
+0.081. On test and on whole recordings they are close to additive (+0.056 and +0.031
alone, +0.096 together). Either way the combination is clearly better than either part, and
it is the largest paired gain over the baseline in the survey.

**It is also the steadiest model yet measured here**: val 0.500 / 0.491 / 0.489 across three
seeds — sd 0.006, against 0.030 for the Hub baseline. Both ingredients reduce variance, as
extra data and richer summaries tend to.

**Fused with the melody branch** (three seeds each, val / test):

| CQT branch | + CREPE histogram | + Essentia histogram |
|---|---|---|
| Hub clips (the released recipe) | 0.508 / 0.440 | 0.537 / 0.407 |
| full recordings | 0.530 / 0.484 | 0.549 / 0.469 |
| 10 clips per performance | — | 0.549 / 0.460 |
| **10 clips + mean/std pooling** | 0.536 / **0.489 ± 0.030** | **0.561 ± 0.012** / 0.478 |

**Chosen on val, as every decision in this survey has been: 10 clips per performance +
mean/std pooling + the Essentia melody branch — val 0.561 ± 0.012, test 0.478 ± 0.023.**
Against the released recipe measured the same way (Hub clips + CREPE, fit on 1350 clips:
0.508 / 0.440) that is **+0.053 val and +0.038 test**, and none of it needed new data.

**Fusion keeps absorbing improvements to the CQT branch.** `clips10_stats` beats `clips10`
by +0.022 val alone but only +0.012 fused; the melody branch was already supplying part of
what the better CQT branch learned. The same pattern held in Batch 10. The two branches
overlap more than their independent-error story suggests, and the fused number moves much
less than either branch does.

**A release candidate, not yet built.** The released model was refit on all 362 train-pool
videos; these runs hold out a fifth. The like-for-like candidate is: 10 clips per
performance from the full recordings, mean/std pooling, refit on all 362 videos, fused with
the Essentia histogram. Expected to land near the released model's 0.480 test plus the
+0.038 measured here. That is a release decision and nothing here forces it.
