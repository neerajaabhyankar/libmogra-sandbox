# survey-aug-2026 — DL raag classification on dataset v1.1

Lab notebook. Newest findings go at the bottom of each stage; **nothing is deleted, including
things that failed** — a negative result costs the same to produce as a positive one and is
worth more than a blank space.

Standing brief and rules: [`CLAUDE.md`](CLAUDE.md). Run registry: [`RUNS.md`](RUNS.md).

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

## Stages

Status legend: ☐ not started · ▶ running · ✔ done · ✖ tried and failed · ⏸ parked

### Stage 0 — harness, caches, and cheap probes
✔ `common/` modules, audio cache (2.34 GB, 0 errors), Sa-anchored and fixed-fmin CQT caches
✔ Frozen-representation probes, video-grouped 5-fold CV — **see the log entry below**

### Stage 1 — retrain as-is on v1.1
☐ D1 distilHuBERT, original recipe, v1.1, grouped val  *(batch 2, ~4 h)*
☐ R1 jeevster ResNet, Stage 6a recipe (last 2 blocks + head), v1.1, grouped val
▶ C1 CQT-ResNet with fixed fmin (absolute pitch) — the control for Stage 2

### Stage 2 — the tonic  *(important)*
☐ D2n / R2n — tonic-normalised audio · D2c / R2c — FiLM conditioning
☐ C2 — tonic-rolled CQT
☐ shuffled-tonic control for each

### Stage 3 — source separation as pre-processing
☐ HPSS (`../source-separation`) in front of the best Stage 2 config per architecture.
  Prior from motif-classifier: separation helped the *tracker* and **hurt classification**
  (M9 0.422 → 0.393), the theory being that HPSS smooths away meend and gamak. A spectral
  model may not care about the same thing a 120-bin histogram cares about, so it is worth one
  run per architecture — but the expectation is set low.

### Stage 4 — the libmogra DB as a prior  *(important)*
☐ P1 musically-graded label smoothing from `raagspace.affinity()`
☐ P2 auxiliary head predicting the DB swar-occupancy vector (multi-task)
☐ P3 fixed DB-template output layer (C only — project to occupancy, score by chi²)
☐ P4 inference-time log-prior fusion with M12's scores
  Prior from motif-classifier: the DB is **a good prior and a poor model** — blending at
  λ=0.3 beat both λ=0 and λ=1 for both M12 and M13. Expect the same shape.

### Stage 5 — hybrid with the symbolic methods  *(stretch)*
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

| run | what | val top-1 | vs its Stage 1 |
|---|---|---|---|
| **c2** | CQT anchored so bin 0 is Sa | **0.302** | **2.7x** |
| r2n | jeevster ResNet, tonic-normalised audio | 0.287 | **2.0x** |
| r1 | jeevster ResNet, as-is | 0.146 | — |
| c1 | CQT, fixed fmin | 0.111 | — |
| c2_shuffled | c2 with tonics permuted between videos | 0.087 | *control* |
| r2c | jeevster ResNet, tonic by FiLM | 0.057 | **0.4x — worse than no tonic** |

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

| run | tonic | val top-1 | mistake affinity (chance 0.260) |
|---|---|---|---|
| d1 | none | 0.080 | 0.266 |
| d2n | normalised into the audio | 0.087 | 0.285 |
| d2c | by FiLM | 0.076 | 0.267 |

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

| run | what the model is told about the tonic | val top-1 |
|---|---|---|
| r2n | it is baked into the audio (normalised) | **0.287** |
| r1 | nothing | 0.146 |
| r2c_shuffled | a *wrong* tonic, by FiLM | 0.130 |
| r2c | the *correct* tonic, by FiLM | 0.057 |

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

| run | | val top-1 | vs its unseparated twin |
|---|---|---|---|
| c3 | CQT + Sa-anchor, HPSS melody stem | 0.304 | c2 0.302 → **+0.002** |
| r3 | ResNet + normalised audio, HPSS melody stem | 0.113 | r2n 0.287 → **-0.174** |

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

| | top-1 | top-5 | video vote | macro-F1 | mistake affinity (chance) |
|---|---|---|---|---|---|
| c2 | 0.302 | 0.589 | 0.370 | 0.257 | 0.380 (0.263) |
| **c4h** | **0.417** | **0.722** | **0.489** | **0.354** | **0.430** (0.262) |

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
