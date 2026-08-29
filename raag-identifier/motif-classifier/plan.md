# motif-classifier: phrase-based raag identification

Classify a raag from audio by **matching the melodic phrases actually sung against the
`mukhyanga` (characteristic phrase) entries in the libmogra raag database** (scraped from
tanarang.com). No learned audio embeddings, no spectral classifiers — the only thing that
touches audio is the note transcription from `../melody-extraction`.

---

## Headline result

> **Superseded by dataset v1 (2026-08-28) — see [v1 of the dataset](#v1-of-the-dataset-the-tonic-stops-being-a-guess) at the end.**
> Everything in this section was measured on v0 (~6 s clips, estimated tonic). It is kept as
> written: the v0 numbers are still the control that the v1 comparisons are measured against.
> Current best is **M9+ at 0.353 test top-1**, against the 0.185 below.

50-way, chance = 0.020. Test = the held-out `test_*` clips, scored once, with the config
grouped-CV on train picked. Full tables + confusion matrices in
[`results/RESULTS.md`](results/RESULTS.md).

| method | idea | train top-1 (CV) | **test top-1** | test top-5 |
|---|---|---|---|---|
| M1 | exact mukhyanga substring | 0.041 | 0.087 (8/92) | 0.207 |
| M2 | n-gram / skip-gram overlap + scale | 0.104 | 0.130 (12/92) | 0.272 |
| M3 | phrase grammar (smoothed bigram LL) | 0.107 | 0.130 (12/92) | 0.315 |
| M4 | **M3 over Tony + CREPE fused** | 0.120 | 0.163 (15/92) | 0.370 |
| M5 | noisy-channel HMM (kan-swar/meend model) | 0.110 | 0.109 (10/92) | 0.261 |
| M6 | joint tonic+raag, learned rotation prior | 0.132 | 0.109 (10/92) | 0.315 |
| **M7** | channel + CREPE + tonic prior + calibration | **0.149** | **0.185 (17/92)** | 0.359 |
| M8 | soft (un-quantized) swar membership | 0.125 | — (folded into M9) | — |
| M9 | melody surface — un-quantized contour, **no mukhyanga** | 0.111 | 0.163 (15/92) | 0.337 |
| **M9+** | **M4 + melody surface** | 0.135 | **0.185 (17/92)** | **0.413** |

**M9+ is the best method overall.** It ties M7's 0.185 top-1 but beats it on everything
softer — top-5 **0.413** vs 0.359, MRR **0.307** vs 0.281, video-vote 0.196 vs 0.152 — and it
is much simpler (no channel HMM, no rotation marginalisation). See the M9 section for why.

For scale: the sibling **supervised** spectrogram ResNet in
`../hindustani-raag-classifier-resnet` tops out at **0.174** on this same 50-class test
split after full fine-tuning. M7 reaches **0.185 with no acoustic model at all** — the raag
knowledge is 50 dictionary entries, and the only things fit on train are ~250 scalars (a
12×12 emission matrix, a 12-way rotation prior, 50 mean/spread pairs, and a handful of
hyperparameters), none of them raag-specific except the last set of offsets. That is the
main result: on this corpus, prescriptive phrase knowledge is worth about as much as a
trained CNN.

> **Caveat on the test split.** It has 92 clips, so one clip = 1.1 points and the M4/M7 gap
> is not significant. Test was also consulted between the two rounds of work (M1–M3, then
> M4–M7), so the M4–M7 numbers are not perfectly blind — the *configs* were chosen on train
> CV throughout, but the decision to build M4–M7 came after seeing M1–M3's test numbers.
> The train-CV column is the honest ranking; treat test as confirmation, not discovery.

---

## Inputs, fixed

| Piece | Source | Notes |
|---|---|---|
| Raag knowledge | `libmogra 0.4.2` → `RAAG_DB` | 117 raags; `aaroha`, `avaroha`, `mukhyanga`, `*_nyas`, `vaadi`, `samvaadi`, `thaat`. All 50 dataset folders map on (`Sarang` → `sarang (brindavani sarang)`, `KaushikDhwani` → `kaushik dhwani (bhinn shadj)` pinned by hand) |
| Audio | `../hindustani-raag-small` | 50 raags, 1161 `train_*` / 92 `test_*` mp3 chunks |
| Transcription | `../melody-extraction` **only** | `tony` (pYIN note HMM) and `crepe` both extracted in full |

Data facts that shape every design choice:

- **Clips are short.** 4–40 s, median ≈ 10 s → a **median of 13 notes** (Tony) or 23 (CREPE).
- **Chunks come in threes from one video.** All folds are **grouped by video id**; otherwise
  "validation" measures recording recall, not raag recognition.
- **`mukhyanga` is sparse.** 2–11 phrases per raag (mean 5.0), median 4 swars each.
- Test covers 46 of the 50 raags; the candidate set stays all 50.

---

## Diagnostics: what is recoverable at all

`diagnostics.py`, on train, with the final representation:

| quantity | value | reading |
|---|---|---|
| notes per clip | median 13 | very little to match against |
| clip duration on the **true** raag's swars | **0.80** | tonic usually roughly right |
| …at the best of the 12 rotations | 0.87 | ~7 pts lost to tonic, ~13 to transcription noise |
| clips containing ≥1 verbatim mukhyanga | **0.20** | M1's hard ceiling |
| mean fraction of a mukhyanga's bigrams present | 0.27 | roughly M2's |

Corroboration from the literature: published work puts automatic pitch-tracker accuracy on
Indian art music at around **80 %**, which is exactly the in-scale number measured here.
The transcription, not the matcher, is the noisy component.

---

## The tonic — the single biggest lever

Everything in the DB is relative to Sa; the transcription is in Hz.

**(a) melody-extraction's heuristic — kept.** Mode of the voiced pitch, snapped to a
semitone, pooled per *video* (chunks of a video are one recording, available side by side at
test time). Per-clip ≈ per-video; per-video kept for stability.

**(b) sub-semitone tuning refinement — kept, small.** The heuristic snaps to the A440
semitone grid, so it can sit 50 cents off the real Sa, biasing **every** note. `refine_tonic`
takes the circular mean of `cents mod 100` over voiced frames and shifts by it. Present in
every winning config; worth ~0.5–1 pt.

**(c) Sa-vs-Pa correction from the chroma histogram — tried, failed.** The rotation
histogram peaks at +7 after 0, consistent with the mode landing on the tanpura's **Pa**.
`chroma_tonic` scores candidates by "has a strong fifth above, is not itself a fifth above
something strong, sits near the bottom of the sung range"; `tune_tonic.py` grid-searches its
four weights. **0.799 → 0.804 in-scale**, and downstream top-1 got slightly *worse*
(0.089 → 0.080). Kept in the code, off by default.

**(d) hard max over rotations — tried, failed.** `shift_mode="global"`/`"per_raag"` lost
badly (M3: 0.089 → 0.074). Twelve chances at a spurious fit cost more than wrong tonics do.

**(e) soft marginalisation with a learned prior — worked (M6/M7).** See below. The prior
learned on train is `P(k=0)=0.29, P(k=7)=0.15, P(k=5)=0.11` — the tanpura-Pa hypothesis is
real, it is just not something to *commit* to per clip.

**And the number that dominates the whole project:** with an **oracle tonic** (rotation
chosen using the true label), M3 goes **0.109 → 0.314 top-1** and **0.277 → 0.721 top-5**
(`ceilings.py`). Sa placement is worth roughly **3× everything else combined**.

---

## Representation

```
mp3 → melody-extraction tracker → note events {t_start, t_end, f0_hz}   [cached, pre-tonic]
    → cents vs tonic → nearest of 12 swar classes → collapse repeats → swar string
```

Chosen by `tune.py --stage rep` (144 configs):

| knob | chosen | notes |
|---|---|---|
| `note_source` | **`hmm`** (Tony's note HMM) | decisive — swept the entire top-12. Routing Tony's *frame* track through `segment_notes` gives 40 % more notes and scores **half** as well (0.089 vs 0.043). More notes ≠ better notes |
| `max_cents_dev` | **2400** | the initial ±1 octave clamp gutted clips whose tonic landed an octave off |
| `min_dur` | **0.0** | every duration filter hurt; `prunethresh` already removed the junk |
| `tonic_mode` / `tonic_refine` | `video` / `True` | see above |
| `collapse_repeats` | `True` | re-swept per method; `True` won for all seven |

Octave marks (`,N`, `` `S ``) are **dropped** — tracker octave errors are common enough that
matching on saptak costs more hits than it buys.

---

## Methods

### M1 — naive exact phrase matcher (baseline)

Fraction of the raag's mukhyanga occurring as a contiguous substring, + a longest-match bonus.

- **Tried:** `length_bonus`, `shift_mode`, `collapse_repeats`.
- **Result:** CV 0.041, test 0.087.
- **Verdict:** fails exactly as the diagnostics predicted — 80 % of clips contain no verbatim
  phrase, so 80 % of decisions are ties broken by nothing. Not a tuning failure; the corpus
  does not contain the thing it looks for.

### M2 — n-gram / skip-gram overlap + scale term

Phrases broken into 2/3-grams; the clip contributes skip-grams stepping over up to
`max_skip` intervening notes; plus a duration-weighted scale term.

- **Tried:** 1728 configs.
- **Result:** CV 0.104, test 0.130.
- **Worked:** partial credit (2.5× M1). Skip-grams (`max_skip` 1–2 > 0). `w_scale > 0` always.
- **Didn't:** **IDF weighting, the headline idea, was tuned away** — every top config chose
  `idf_power = 0.0`. With so few hits per clip, down-weighting common n-grams discards the
  only evidence there is. `len_power = 0` likewise: long phrase fragments aren't more
  trustworthy, because long phrases are what the tracker mangles.
- **Surprise:** `w_arohana = 1.0` — the plain aaroha/avaroha are worth **as much as the
  mukhyanga**. The scale-and-direction skeleton survives transcription noise; ornate
  characteristic phrases don't.

### M3 — phrase grammar as a generative sequence model

Phrases + aaroha/avaroha → smoothed bigram transition model over 12 swars; score = LL of the
clip's swar sequence. Every transition is evidence, so a clip with no verbatim motif is still
classifiable.

- **Tried:** 5184 configs.
- **Result:** CV 0.107, test 0.130, **top-5 0.315**.
- **Worked:** `w_skip = 0.5` (note-pairs one apart alongside adjacent pairs) — in essentially
  every top config, the most consistent single win. `w_dur ≥ 0.5`. `uni_from_scale = 0.75`,
  i.e. the smoothing leans heavily on plain scale membership.
- **Didn't:** `dur_weighted` (weighting each *transition* by its shorter note) — nothing,
  though weighting the *unigram* by duration helped. Trigrams never made the shortlist (DB
  phrases are too short to estimate them). `symmetric` was a wash.

### M4 — CREPE, and Tony + CREPE fused  *(lever (a))*

Weighted sum of the two trackers' length-normalised grammar log-likelihoods.

- **Tried:** `w_crepe` ∈ {0…2}, and each tracker alone.
- **Result:** CV 0.120, **test 0.163**, top-5 0.370. Best `w_crepe = 1.0` — equal weight.
- **Worked, and this was the surprise of round 2.** CREPE *alone* scores **0.102** CV,
  essentially level with Tony's 0.109, and fusing them gains **+0.011 CV / +0.033 test** over
  either. The two trackers make genuinely different errors: Tony's note HMM prunes fast
  material, CREPE keeps it and adds noise. Averaging cancels part of both.
- **Note this corrects an earlier conclusion.** The rep sweep found Tony's *frames* through
  `segment_notes` were much worse than Tony's note HMM, and I read that as "frame-level
  segmentation is bad". Wrong: CREPE's frames through the same segmenter are fine. It was
  Tony's smoothed track specifically that segmented badly.
- **Also the best method by mistake quality** — see below.

### M5 — noisy-channel HMM  *(lever (b): kan-swars and meends)*

The DB phrase and the transcription are not two samples of the same alphabet: one is what was
*intended*, the other is that through a noisy channel. A meend P→S drags through G and R and
the segmenter writes `P G R S` for the phrase `P S`.

```
hidden h_t = intended swar   transitions A_r = (1-p_self)·grammar(raag) + p_self·I
observed o_t = written swar  emissions   E   = P(written | intended), LEARNED
score = forward algorithm
```

`E` is estimated on train by **Baum–Welch with each clip's transitions pinned to its true
raag** — a model of the tracker's ornament behaviour pooled over all raags, not of any raag.
Nothing about which substitutions are plausible is hand-written. Refit per CV fold.

- **Tried:** 234 configs incl. a hand-set-channel ablation.
- **Result:** CV 0.110, test 0.109.
- **Verdict as a standalone method: a wash, or slightly negative** — CV 0.110 vs M3's 0.107,
  and inside M7 the channel alone scores 0.104 against plain M3's 0.109. `learn_emissions=True`
  did beat the hand-set channel in every top config, so the *learning* works; the smeared
  emissions just cost as much discriminative power as the ornament tolerance buys. Its real
  value shows up only under tonic marginalisation (M6, M7) — see M7's ablation table.
- **The learned channel is the most interesting artefact in the project**, independent of
  its accuracy contribution:

  | | |
  |---|---|
  | diagonal mass | **0.33** — only a third of notes survive as intended |
  | ±1 semitone mass | 0.24 — semitone confusion dominates, exactly the kan-swar / meend / shruti story |
  | **every row leaks into `S`** | `m`→S 0.20, `P`→S 0.22, `D`→S 0.17, `M`→S 0.16 — **tanpura bleed**: pYIN periodically locks onto the drone |
  | `d`→`D` 0.18 vs `d`→`d` 0.19 | komal and shuddha dhaivat are **indistinguishable** under 12-TET rounding — the shruti problem this whole repo exists for, measured |

  That table alone is an argument for shruti-aware quantization and for source separation.

### M6 — joint tonic + raag  *(lever (c))*

`P(raag|clip) ∝ Σ_k P(k)·P(clip | raag, rotated k)`, with `P(k)` learned on train from the
posterior-weighted rotation frequency under the *true* raag. The two earlier behaviours are
endpoints of this one: a prior on k=0 reproduces `shift_mode="none"`, a flat prior at T→0
reproduces the failed hard max.

- **Tried:** 42 configs over base ∈ {M3, M5}, temperature, `learn_prior`.
- **Result:** CV **0.132** (best single-lever method), test 0.109.
- **Worked on train:** soft marginalisation clearly beats both endpoints (M3 alone 0.107, hard
  max 0.074). `learn_prior=True` > `False` (0.135 vs 0.125). Best base is **M5's channel**,
  not M3 — the emission smearing makes the per-rotation likelihoods better calibrated, which
  is exactly what marginalisation needs.
- **Didn't transfer to test** (0.109). With 92 clips the CV/test gap is ~1.5 clips of noise,
  but the honest reading is that M6's *argmax* is erratic even where its ranking is good — its
  mistake affinity is the worst of the seven (0.271 vs 0.259 chance) while its NLL is
  second-best. Marginalising scrambles the top-1 and improves the ranking.

### M7 — combination + hubness calibration

M5's channel + CREPE fusion + M6's tonic marginalisation + one new piece:

**Per-raag score calibration.** The M2/M3 error analysis showed a few raags predicted far
more often than they occur (AheerBhairav, KaushikDhwani, Shree, Pilu took 26 of 92 test
predictions). That is a *hub* problem, not a music problem — those raags' models score
everything highly because their scales are large or their grammars flat. Standardising each
raag's score against the mean/spread it produces over **train** clips removes the offset
without touching the ranking within a raag.

- **Result:** CV **0.149**, **test 0.185 (17/92)**, MRR 0.281 — best on every aggregate.
- **Full 16-way ablation (CV top-1).** This is more interesting than "all four help":

  | channel | CREPE | calibrate | marginalise | top-1 |
  |:-:|:-:|:-:|:-:|---|
  | – | – | – | – | 0.109 (= plain M3) |
  | ✓ | – | – | – | 0.104 |
  | – | ✓ | – | – | 0.122 |
  | ✓ | ✓ | – | – | 0.121 |
  | – | ✓ | ✓ | – | 0.138 |
  | ✓ | ✓ | ✓ | – | 0.137 |
  | ✓ | – | – | ✓ | 0.135 |
  | – | ✓ | ✓ | ✓ | 0.132 |
  | **✓** | **✓** | **✓** | **✓** | **0.150** |

  **CREPE fusion (+0.013) and hubness calibration (+0.016) are the two reliable
  contributors** — they help with or without anything else. The channel HMM on its own is
  *worse* than plain M3 (0.104 vs 0.109) and adds nothing to CREPE+calibration
  (0.137 vs 0.138). It earns its place only in combination with tonic marginalisation:
  0.132 without it, **0.150** with. That is the same thing M6 found independently (its best
  base was M5's channel, not M3's grammar) and it has a clean explanation — the emission
  smearing is what makes the twelve per-rotation likelihoods comparable enough to
  marginalise over. **The channel is not an accuracy trick, it is a calibration device.**

### M8 — soft (un-quantized) swar membership  *(the weak form of "stop quantizing")*

The last round's headline recommendation was "stop rounding to 12 swars". The cheapest way
to act on that is to keep the DB grammar and make the *observation* soft: a note at 175
cents is not `R`, it is 0.76 `R` + 0.22 `r`, and every count (unigram, bigram, skip-bigram,
and the channel HMM's evidence term) accumulates over that membership instead of a hard bin.
`sigma -> 0` recovers the old behaviour exactly, so this is a strict generalisation.

Two related knobs were tried alongside it:

- **shruti quantization** — use `libmogra.datatypes.SWAR_BOUNDARIES` instead of 12-TET
  rounding. **Dead end, and worth recording why:** those boundaries turn out to sit within
  ~5 cents of the equal-tempered midpoints (45.7, 147.5, 249.6, … vs 50, 150, 250, …). The
  table does not encode meaningfully uneven divisions, so the option is a no-op. My earlier
  "SWAR_BOUNDARIES already encodes non-equal boundaries and was not used" was wrong.
- **learned per-swar tuning offsets** — where the corpus actually sings each swar. Measured
  against the *true raag's scale* (assigning to the nearest of 12 shrinks the estimate to
  ±3 cents by construction, which is an artefact): `r +7, g +6, n +6, m +6, M +6, G −8`.
  The komal swars are sharp and shuddha G is flat, which is the right direction and
  just-intonation-shaped — but the magnitudes are small, partly because `tonic_refine`
  has already absorbed the corpus-wide part of the deviation.

- **Result (CV, on M4):** `sigma` 0 → **0.124**, 25 → 0.125, 40 → 0.111, 55 → 0.068.
- **Verdict: it does not work.** A tiny, noise-level gain at `sigma = 25` cents and a
  collapse beyond it. In hindsight the reason is obvious: raags that differ by exactly one
  komal/shuddha swap are a large fraction of the confusable pairs, and blurring a note
  across neighbouring swars is precisely the operation that erases that distinction.
  Un-quantizing by *widening the bins* destroys more than it recovers.

### M9 — time-delayed melody surfaces  *(the strong form, and it works)*

If quantization is the problem, do not quantize at all. Following Gulati et al.'s melody
surface, represent a clip by the 2D histogram of (pitch at `t`, pitch at `t + tau`) taken
straight from the **frame-level f0 track** — no note segmentation, no 12-bin rounding —
octave-folded at 20–30 cents per bin. The diagonal is the pitch distribution at shruti
resolution; the off-diagonal ridges are transitions, and their *width* is the meend.

The price: a mukhyanga cannot be written as a surface, so **M9 uses no database at all** —
each raag's reference is the mean surface of its train clips. It is the honest measure of
what the pitch space gives you without the prior.

- **Tried:** 10 configs — tracker, `n_bins` ∈ {40,60,80}, `tau` ∈ {0.15,0.3,0.5,0.6},
  smoothing, power compression, cosine vs L2.
- **Result:** CV **0.111**, **test 0.163 (15/92)**, top-5 0.337 (CREPE, `n_bins=60`,
  `tau=0.3`) — the same test top-1 as M4, **with no access to the database at all**.
- **The tracker choice is the whole story: CREPE 0.103 vs Tony 0.045.** Tony's note HMM has
  already smoothed and quantized the contour, so there is no shruti or meend detail left for
  the surface to see. The representation that beat everything else at note level is the
  worst possible input here — a nice illustration that "best tracker" is not a property of
  the tracker but of the tracker-plus-representation pair.
- **Context for 0.103:** the entire mukhyanga database, used well (M3), gets 0.107. A
  bigram LM learned from the same transcriptions gets 0.073. So **the un-quantized contour
  carries about as much raag information as all of tanarang.com** — which is the strongest
  evidence yet that quantizing to 12 swars is where the loss is.

### M9+ — melody surface fused with the phrase grammar

The two are near-orthogonal by construction: M9 knows nothing about mukhyanga, M1–M7 know
nothing about sub-semitone pitch. Z-score each over train, add.

- **Result:** CV 0.124 → **0.135**; **test 0.163 → 0.185 (17/92)**, top-5 0.370 → **0.413**,
  MRR 0.256 → **0.307**, video-vote 0.130 → **0.196**. Best `w_tdms = 0.5`.
- **This is the best method in the project**, tying M7's top-1 and beating it on every softer
  metric, from far less machinery.
- **Verdict: the un-quantized representation is a genuine additional evidence stream, not a
  replacement.** Fusing helps; swapping does not. Combined with M8's failure, the lesson is
  that the fix for quantization is *another view of the same audio*, not a fuzzier version
  of the existing view.
- Fusing with M7 rather than M4 was started and abandoned on runtime — M7 refits a channel
  HMM and marginalises 12 rotations per clip per fold, and the fusion adds a full extra pass
  over train inside `fit()`. Worth doing with cached score matrices; **not measured here.**

---

## How bad are the mistakes?  (`raagspace.py`, `musical_eval.py`)

Top-1 scores `TilakKamod → Des` exactly as badly as `TilakKamod → Bairagi`. So there is now a
raag-to-raag **affinity** built purely from the DB — TF-IDF cosine over mukhyanga/aaroha
n-grams (following `../../raagspace.ipynb`), swar-set Jaccard, and thaat identity — plus a
separate **rotational** affinity (best match after transposing the predicted scale).

It reproduces musical intuition unprompted:

```
TilakKamod  -> Des 0.64, Khamaj 0.62, Tilang 0.51        (vs Bairagi 0.16)
Bhoopali    -> Deshkar 0.60, Yaman 0.52, HansDhwani 0.43 (vs Bhairav 0.16, Madhukauns 0.12)
Bageshree   -> Bheempalasi 0.75, Kafi 0.73
Todi        -> Multani 0.74
```

Your Bhoop example scores exactly as you'd want: Durga 0.38 / HansDhwani 0.43 / Deshkar 0.60
vs Sohani 0.18 / Bhairav 0.16 / Madhukauns 0.12. **And the one case where direct affinity
disagreed with you is the interesting one** — you called Malkauns a reasonable Bhoop
confusion; direct affinity says 0.05 (they share only S). The *rotational* affinity says
**1.000 at k=4**: Malkauns is Bhoopali's pitch set transposed. Both readings are correct and
they mean different things, which is why the two are reported separately.

Metrics (test; each next to its chance value; softmax temperature calibrated on **train**):

| method | mistake affinity | (chance) | MEA ↑ | (chance) | affinity CE ↓ | NLL ↓ |
|---|---|---|---|---|---|---|
| M1 | 0.295 | 0.261 | 0.279 | 0.274 | 3.860 | 3.826 |
| M2 | 0.291 | 0.257 | 0.296 | 0.274 | 3.804 | 3.748 |
| M3 | 0.290 | 0.257 | 0.286 | 0.274 | 3.866 | 3.835 |
| **M4** | **0.333** | 0.259 | 0.301 | 0.274 | 3.807 | 3.758 |
| M5 | 0.287 | 0.258 | 0.285 | 0.274 | 3.869 | 3.842 |
| M6 | 0.271 | 0.259 | 0.307 | 0.274 | 3.791 | 3.612 |
| M7 | 0.287 | 0.261 | 0.323 | 0.274 | **3.763** | **3.550** |
| M9 | 0.306 | 0.260 | 0.331 | 0.274 | 4.335 | 4.050 |
| **M9+** | 0.327 | 0.259 | **0.333** | 0.274 | 3.877 | 3.728 |

- **`mistake_affinity`** — affinity(true, predicted) over errors only.
- **`MEA`** — `Σ_r p(r)·affinity(true,r)` over the whole softmax output. This is the metric
  that rewards your Bhoop example.
- **`affinity_ce`** — cross-entropy against a soft target `q ∝ affinity(true,·)^4`. The
  mukhyanga-based loss: it does not punish mass on genuinely related raags. Chance = ~3.91.

Readings:

1. **Every method's mistakes are better than random, but only slightly** — +0.03 affinity
   over chance for most, i.e. the errors are ~12 % more musically sensible than a coin flip.
   Sobering. The models are not just picking a plausible neighbourhood and missing.
2. **M4 (tracker fusion) has by far the most principled mistakes**: 0.333 vs 0.259 chance,
   +29 % — more than double any other method's margin. When it misses it misses *close*.
3. **M9+ has the best expected affinity (0.333) and the second-best mistake affinity
   (0.327).** Adding the continuous contour makes the mistakes *more musical*, not just
   fewer — which is what you would hope from a representation that finally sees shruti.
4. **M7 stays best on the two likelihood-shaped losses** (affinity-CE 3.763, NLL 3.550).
5. **M9 alone is badly calibrated**: NLL 4.050 is *worse than chance* (3.912) even after
   temperature scaling, because a cosine between surfaces does not behave like a
   log-probability. Its ranking is good (MEA 0.331) but its confidences are not usable
   as-is — fusing it (M9+) fixes that, and anything reading M9's scores as probabilities
   should calibrate them properly first.
5. **`tonic_explained` sits near chance for all methods.** Combined with the oracle result,
   the conclusion is precise and slightly surprising: a wrong tonic *destroys the true raag's
   score*, but the raag that wins instead is **not** systematically the true one transposed —
   it is noise. So "detect and undo the rotation post-hoc" will not work; the tonic has to be
   right before scoring.

Sample errors from M7 (test):

```
most defensible   PuriyaDhanashri -> Basant 0.75 · TilakKamod -> Des 0.64 · MaruBihag -> Kedar 0.59
least defensible  Sohani -> Dhani 0.05 (rot 0.83 @k=6) · Bhoopali -> Multani 0.09 (rot 0.71 @k=1)
```

---

## Is there hope in the symbolic space? (`ceilings.py`)

Six experiments, grouped-CV on train, each isolating one constraint.

| # | experiment | result |
|---|---|---|
| 1 | **oracle tonic** (rotation picked with the true label) | 0.109 → **0.314** top-1, 0.277 → **0.721** top-5 |
| 2 | **accuracy vs. notes per clip** | ≤8 notes **0.065** → 8-14 0.095 → 14-22 0.128 → 22-35 0.140 → ≥35 **0.184** |
| 3 | **pool a video's 3 chunks** (3× notes, no new data) | 0.109 → **0.149** (median 13 → 39 notes) |
| 4 | **data-driven only** (bigram LM from transcriptions, no mukhyanga) | **0.073** vs DB-only 0.109 |
| 5 | **DB + data-driven** | 0.111 — no gain over DB alone |
| 6 | **learning curve** (data-only, by fitting videos) | 76 vids 0.043 · 152 **0.075** · 229 0.065 · 305 **0.073** — **flat after ~150 videos** |

**The answers:**

- **Yes, but the ceiling is the tonic, not the method.** Experiment 1 is unambiguous: Sa
  placement is worth 3× everything else. Every hour spent on phrase matching is competing
  with an available 20-point gain from tonic estimation.
- **Clip length is the second constraint, and it is nearly as big.** Experiments 2 and 3
  agree: 3× the notes buys ~1.4× the accuracy, and the trend has not flattened at 35 notes.
  Ten-second clips are simply below the amount of melody a raag needs to declare itself.
- **More training data will *not* help at this transcription quality.** Experiment 6 is flat
  from 150 videos on. The data-driven model is noise-limited, not data-limited — doubling the
  corpus would double the noise along with the signal. This is the clearest possible answer
  to "(a) increase training data": **not yet**. Fix the pitch pipeline first.
- **The mukhyanga database is genuinely carrying the result.** Experiment 4: prescriptive
  knowledge (0.109) beats everything learnable from 1161 clips (0.073), and experiment 5 says
  the data adds nothing on top. At this scale the DB is not a crutch, it is the model.

### Where this sits against published work

The closest published analogue is **Gulati et al., "Phrase-based rāga recognition using
vector space modeling" (ICASSP 2016)** — melodic patterns discovered unsupervised, TF-IDF
over a pattern vocabulary, i.e. this project done at scale. It reports **~70 % on 40 rāgas**
(92 % on a 10-rāga subset) — but over **480 recordings totalling 124 hours**, roughly 15
minutes per item against our 10 seconds, i.e. **~90× more melodic material per decision**.
Given experiment 2's length curve, that gap alone plausibly explains most of the difference.

The closest *short-clip* result is the TISMIR multimodal work: **86.2 % on 9 rāgas from 12-s
clips**, with a pitch-only audio baseline at **84.3 %** — supervised, and 9 classes rather
than 50. Reported Hindustani numbers of 98–99 % (pitch co-occurrence features, PhonoNet) are
all on full-length recordings and smaller/curated rāga sets.

So: this setting — **50 classes, 10-second clips, no acoustic training** — is the hard corner
of all three axes, and 0.185 is not embarrassing there. But nothing in the literature suggests
50-way from 10 seconds is a solved or nearly-solved problem.

### What is left to try in the symbolic space

Ordered by expected value. The first item is the big one, and it is a criticism of a choice
made on line one of this project.

1. **Stop quantizing to 12 swars.** Every method here turns the pitch curve into a string
   over a 12-symbol alphabet, and the two families that actually work well on pitch-space
   raga ID deliberately do not: **time-delayed melody surfaces** (Gulati et al., the
   companion to the VSM paper) and **sequential pitch distributions** both operate on the
   *continuous* contour — a 2D histogram of pitch at `t` against pitch at `t+τ`, at
   finer-than-semitone resolution. That representation keeps exactly what we throw away:
   the shape of the meend, the gamak width, and the shruti. Our own learned emission matrix
   is direct evidence of the cost — `d`→`D` confusion at 0.18 against a diagonal of 0.19
   means 12-TET rounding is destroying a distinction the raag depends on. This is still
   "symbolic / no audio": the input is the same `f0` track `melody-extraction` already
   produces. **This is the highest-value untried item in the whole project.**
2. **Rotation augmentation.** The tonic is the dominant error (oracle: 3×) and a purely
   symbolic fix exists: train/fit on all 12 transpositions of every train sequence so the
   model is tonic-robust by construction, rather than trying to get Sa right first. Cheap,
   and it composes with everything already built.
3. **Pretrain a sequence model on a bigger symbolic corpus.** A transformer/LSTM over swar
   sequences is hopeless from 15k notes (experiment 6 shows even a bigram LM saturates), but
   this repo already holds `../hindustani-raag-fullaudios`, `../../transcriptions/` and
   `../../data-dunya-hindustani/`. Transcribe those, pretrain, fine-tune on the 50 classes.
   This is the route that would let go of the mukhyanga dictionary entirely — but note
   experiment 4 says the DB currently beats anything learned, so this only pays after (1).
4. **Discriminative model over n-gram features.** We built a *generative* bigram LM;
   logistic regression / linear SVM over TF-IDF n-gram features is the Gulati VSM recipe and
   is usually stronger at the same data scale. Straightforward given `features.py`.
5. **Richer alphabet before richer model.** Octave-aware symbols (with a ±1-octave penalty
   rather than folding) and duration-bucketed symbols both add information we currently
   discard, and neither needs new data.
6. **Local alignment (Smith–Waterman)** over the swar string, to distinguish "one phrase,
   stretched" from "three unrelated fragments" — M2's bag-of-n-grams cannot.

**Do we have hope? Yes, but not by tuning matchers.** The three measured constraints are, in
order: Sa placement (worth ~20 points), melodic material per clip (worth ~8 points within the
corpus, more with full recordings), and pitch quantization (unmeasured here, but the emission
matrix says it is real). Dataset *size* is measurably **not** a constraint yet. So the honest
ordering of your three options is: **(c) fix the representation → (b) longer audio per item →
(a) more clips**, with (a) last.

---

## Recommended next directions

**Best two methods, and why they are the two:**

- **M9+** — the one to build on. Test top-1 0.185, top-5 **0.413**, MRR **0.307**, best MEA,
  second-best mistake affinity, and structurally the simplest of the top group: an M3 grammar
  over two trackers, plus one melody surface, z-scored and added. Its two halves are
  near-orthogonal, which is exactly why it has room left.
- **M7** — same top-1, best calibrated (lowest NLL and affinity-CE), and the only method that
  does anything about the tonic. Keep it for the tonic marginalisation, which M9+ lacks
  entirely and which the oracle says is where the remaining 20 points live.
- **M4** still has the single best mistake affinity (0.333) and is the cheapest to run; it is
  the right baseline to iterate against.

Then, in priority order:

1. **Tonic estimation, as its own project.** The 0.109 → 0.314 oracle gap dwarfs everything
   else here. Two concrete routes, neither of which was tried: (i) estimate the tonic from the
   **tanpura drone** directly — it is the most stable, most continuous pitch in the mix and
   the chroma-histogram approach failed because it looks at the *melody*, not the drone;
   (ii) pool the tonic estimate over the **whole source video** rather than a 30-second window.
2. **Longer sequences.** Experiment 3 gets +0.04 for free by concatenating chunks. The full
   `../hindustani-raag-fullaudios` recordings exist in this repo and would give minutes rather
   than seconds — the single cheapest large gain available, and the fair comparison against
   the Gulati-style literature.
3. **Shruti-aware quantization.** The learned channel measures the problem exactly: `d`→`D`
   confusion at 0.18 against a `d`→`d` diagonal of 0.19. `libmogra.datatypes.SWAR_BOUNDARIES`
   already encodes non-equal boundaries and was not used — 12-TET rounding is throwing away
   precisely the distinction this repo was built to study.
4. **Source separation before pitch tracking.** Every row of the emission matrix leaks into
   `S`, which is the drone. Removing the tanpura should improve both the transcription *and*
   (with route 1) give a clean tonic — one fix, both problems.
5. **Tracker ensembling beyond two.** M4's gain came from disagreement between two trackers.
   `praat` and `pyin` are already implemented in `../melody-extraction` and untested here.
6. **Alignment instead of bags.** M2 counts n-grams in a bag; Smith–Waterman over the swar
   string with gap penalties would distinguish "one phrase, stretched" from "three unrelated
   fragments". Genuinely different from anything tried, and cheap.

Not recommended yet: **more training data** (experiment 6 says it is not the constraint), and
**going back to audio embeddings** (the supervised ResNet already did, and lost).

---

## Layout

```
motif-classifier/
  plan.md            # this file
  raagdb.py          # libmogra DB -> folder mapping, folded phrases, scale sets, n-gram IDF
  extract.py         # batch transcription via ../melody-extraction -> cache/notes_<tracker>.npz
  represent.py       # cached notes (+ tonic policy) -> swar sequences
  features.py        # swar sequence -> n-gram / bigram / duration stats, rotatable; MultiFeatures
  diagnostics.py     # transcription-quality ceilings (run before believing any number)
  raagspace.py       # raag-to-raag affinity + rotational affinity, from the DB alone
  musical_eval.py    # mistake affinity, MEA, affinity-CE, tonic-explained
  confusion.py       # 50x50 confusion matrix PNGs
  ceilings.py        # oracle tonic, length scaling, pooling, data-driven baselines, learning curve
  methods/           # m1_exact m2_ngram m3_grammar m4_fusion m5_channel m6_jointtonic m7_combo
  evaluate.py        # metrics + grouped-by-video folds + method registry
  tune.py            # staged grouped-CV sweeps -> results/sweep_*.json
  tune_tonic.py      # Sa-vs-Pa weight search (negative result)
  report.py          # best-on-train config -> single test pass -> results/RESULTS.md + PNGs
  cache/  results/
```

Reproduce:

```bash
poetry run python extract.py --tracker tony      # ~14 min
poetry run python extract.py --tracker crepe     # ~72 min
poetry run python diagnostics.py
poetry run python tune.py --stage rep            # then m1 m2 m3 m4 m5 m6 m7
poetry run python report.py                      # test numbers + confusion matrices
poetry run python report.py --metrics-only       # rebuild tables from results/final.json
poetry run python ceilings.py
```

## Out of scope (still)

- Any audio feature that isn't a note transcription from `../melody-extraction`.
- Training an acoustic model. The only things fit on train are hyperparameter scalars, M5's
  shared emission matrix, M6's rotation prior, and M7's per-raag offsets — all refit per CV
  fold, none raag-specific except the last.
- Talas/laya, ornament classification, octave-aware matching.

---

# v1 of the dataset: the tonic stops being a guess

Everything above this line was measured on **v0** — 1253 clips of ~6 s, tonic estimated from
the audio. On 2026-08-28 the dataset shipped **v1**, and it changed two things at once:

| | v0 | v1 |
|---|---|---|
| clips | 1161 train / 92 test | 1810 train / 150 test |
| clip length | ~6 s | 20-60 s |
| audio | 2.1 h | 11.8 h |
| videos | 432 | 412 (20 dropped, 6 added) |
| tonic | estimated | **hand-annotated, per recording** |

Two changes at once is a confound, so they are separated below rather than reported as one
number. Results live in [`results/v1/`](results/v1); v0's stay in [`results/`](results)
and `RAAG_DATA_VERSION=v0` reproduces them.

**Loading note.** The v1 audio exists only in the parquet files. The `<Raag>/*.mp3` tree at
the repo root is still v0, and the `metadata.csv` the dataset card promises beside it does
not exist — a loader that reads the raw layout silently gets v0 while believing it has v1.
`materialize` from parquet; `audio.path` still carries `{split}_[{video}]_chunk{n}.mp3`,
which is what grouped-by-video CV needs.

## How good was the tonic heuristic, really?  (`tonic_audit.py`)

Until now the estimator could only be judged by downstream accuracy. With ground truth it
can be scored directly. Every method here is octave-folded, so **the error that matters is
mod 12** — an octave error costs nothing.

| tonic policy | pitch-class exact | off by a 4th | abs. within 50c | fine tuning (MAD) |
|---|---|---|---|---|
| per-video, refined | **0.619** | 0.11 | 0.190 | 3c |
| per-clip, refined | 0.523 | 0.13 | 0.163 | 4c |
| per-video + Sa-vs-Pa correction | 0.474 | 0.27 | 0.138 | 3c |
| per-clip + Sa-vs-Pa correction | 0.429 | 0.21 | 0.123 | 4c |

1. **The heuristic gets Sa's pitch class right 62 % of the time.** The 19 % "absolute"
   figure is misleading: 43 % of its errors are pure octave errors, invisible downstream.
2. **When it finds the right pitch class its tuning is excellent — 3 cents.** The failure is
   entirely categorical (*which* scale degree is Sa), never fine. So the annotation's value
   is concentrated in the ~38 % of clips where the heuristic picks the wrong degree, not
   spread thinly over all of them.
3. **The Sa-vs-Pa correction is actively harmful** — 0.619 → 0.474 — and it fails in a
   specific way: it roughly *doubles* fourth-errors (219 → 534 clips landing a fifth below
   true Sa). Built to fix Sa/Pa confusion, it manufactures it in the other direction.
   Every tuned config had already settled on plain `video`, so no earlier result depends on
   it, but `Params.tonic_mode` still defaults to `chroma_video` and **that default is wrong**.

## What the tonic buys  (`tonic_ablation.py`)

Same v1 audio, same tuned configs, only the tonic policy moving. Train CV.

| method | heuristic | + Sa-vs-Pa | **annotated** | tonic gain |
|---|---|---|---|---|
| M3 phrase grammar | 0.167 | 0.135 | 0.271 | +62 % |
| M4 Tony + CREPE | 0.170 | 0.134 | 0.283 | +66 % |
| M7 channel + calibration | 0.214 | 0.146 | 0.329 | +54 % |
| M9 melody surface | 0.192 | 0.159 | **0.340** | **+77 %** |
| M9+ M4 + surface | 0.210 | 0.172 | **0.360** | +71 % |
| M10 emphasis (new, below) | 0.089 | 0.076 | 0.153 | +72 % |

And the two v1 changes separate cleanly, on M3:

* **clip length** (v0 → v1, heuristic held fixed): 0.107 → 0.167
* **tonic** (heuristic → annotated, audio held fixed): 0.167 → **0.271**

The tonic is worth more than every modelling idea in this project put together. M3 — the
simplest DB method — beats v0's best method (M9+, 0.135) by a factor of two, purely by being
told where Sa is.

**M9 gains the most (+77 %).** That fits: a continuous pitch surface is far more sensitive to
a wrong rotation than a smoothed bigram grammar, which can absorb some of the error.

> **Bug found by this ablation.** M9's first run returned *bit-identical* numbers under all
> three policies. `TDMS` builds its surface from a second tracker's frames and so looks the
> tonic up itself rather than taking the clip's — and that lookup hardcoded
> `_load(self.tracker, "video", ...)`, making the method structurally blind to the
> annotation. It now takes `tonic_mode`. The invariance was the tell: three policies that
> disagree on 40 % of clips cannot produce identical accuracy.

## Is the tonic finished?  (oracle headroom)

`ceilings.oracle_tonic` picks each clip's rotation using the true label — cheating, so it
bounds what *any* tonic work could buy.

| | honest | oracle rotation | headroom |
|---|---|---|---|
| heuristic tonic | 0.174 | 0.348 | +0.174 |
| **annotated tonic** | 0.275 | 0.352 | **+0.077** |

The annotation captures **58 % of the available headroom**. The ceiling barely moves
(0.348 → 0.352), which is the sanity check that matters: the oracle maximises over all 12
rotations and therefore should not depend on where it starts. The residual +0.077 is
annotation error, within-video tonic drift, and some pure oracle optimism.

## M10 — emphasis and register  *(the method the tonic unlocks — and it failed)*

Auditing the DB turned up two signals that every method had been discarding:

* `vaadi`/`samvaadi` are parsed and used by exactly one method (M2), which puts them in the
  same **binary** mask as `nyas` — so Bhoopali (vaadi G, samvaadi D) and Deshkar (vaadi D,
  samvaadi G) get *identical* templates despite sharing a scale and differing in nothing
  else. Ranking the emphasis should separate them.
* `phrase_octaves` and `aaroha_oct` are parsed and used **nowhere**.

Both are claims about where weight sits *relative to Sa*, so both were unusable under an
estimated tonic — which is what makes this the method that could not have been written
before v1.

**Result: it does not work, and the failure is informative.** There are 8 scale-sharing
groups covering 20 raags (`scale_twins.py`). On those clips specifically:

| | twin-clip top-1 | confuses into a twin | all-clip top-1 |
|---|---|---|---|
| M3, annotated tonic | **0.240** | 0.131 | 0.272 |
| M10, annotated tonic | 0.140 | 0.113 | 0.151 |

M10 is *worse on the very pairs it was built for*, and its twin accuracy sits below its own
overall accuracy, so it does not find them easier either. Fused with M3 it adds nothing and
hurts monotonically — 0.271 / 0.271 / 0.266 / 0.244 / 0.220 at weights 0 / 0.25 / 0.5 / 1 / 2.
Best weight is zero.

Why: M3's unigram term is already built from scale plus phrase statistics with a
`nyas_boost`, so emphasis is **already in there** — M10 is redundant, not complementary. The
register half was doomed separately: the DB marks too few octaves to discriminate anything
(every raag comes out ~77 % madhya, median pairwise template distance 0.108 against 1.125
for emphasis). *Musicians describe Deshkar vs Bhoopali as a register difference; the database
does not encode it.* That gap, not the method, is the finding.

## What the annotation makes obsolete

Two pieces of machinery built to work around a bad tonic stop earning their place:

* **M7's rotation marginalisation.** Re-tuned on v1, `marginalise_tonic: False` wins
  (0.329 vs 0.292). The 12-way marginalisation existed only because the tonic was unknown.
* **M7's channel HMM** is now a rounding error: 0.329 with it, 0.326 without.
* **M6** (joint tonic + raag, learned rotation prior) is moot by construction.

Conversely, one representation knob flipped: **narrowing to ±1 octave now costs 10 points**
(0.273 → 0.172), where on v0 it barely mattered. Under a wrong tonic the outer octaves were
junk; under the right one they are real melodic material. Otherwise the v0 representation
choices transfer unchanged (Tony's note HMM, `min_dur` irrelevant, `collapse_repeats` inside
the noise).

## v1 headline: the held-out test split

Re-tuned on v1 train with the annotated tonic, then a single pass over the 150 held-out
clips. Full tables in [`results/v1/RESULTS.md`](results/v1/RESULTS.md).

| method | train top-1 (CV) | **test top-1** | test top-5 | test MRR | by video-vote |
|---|---|---|---|---|---|
| M3 phrase grammar | 0.285 | 0.227 (34/150) | 0.607 | 0.385 | 0.420 |
| M4 Tony + CREPE | 0.291 | 0.253 (38/150) | 0.580 | 0.416 | 0.420 |
| M7 channel + calibration | 0.332 | 0.273 (41/150) | 0.607 | 0.437 | 0.500 |
| M9 melody surface (**no** mukhyanga) | 0.338 | 0.340 (51/150) | 0.647 | 0.482 | 0.480 |
| **M9+ M4 + melody surface** | **0.381** | **0.353 (53/150)** | **0.673** | **0.498** | **0.540** |
| _v0's best (M7 / M9+)_ | 0.149 | 0.185 (17/92) | 0.359 | 0.281 | 0.152 |
| _supervised spectrogram ResNet (v0)_ | — | 0.174 | — | — | — |
| _chance_ | 0.020 | 0.020 | 0.100 | 0.090 | 0.020 |

**M9+ nearly doubles v0's best: 0.353 against 0.185**, at 17.6x chance. The median rank of
the true raag is **3 of 50**, and 33 of the 50 raags are correct at rank 1 on at least one
clip. Two thirds of the gain is the tonic, one third the longer clips.

The ranking also reshuffled: **M9 alone (0.340) now beats M7 (0.273)** — with no mukhyanga,
no phrase grammar, no database at all beyond the candidate list. Give the un-quantized
contour a correct Sa and it outperforms every knowledge-driven method here.

Calibration improved too: M9+'s NLL is 2.817 against 3.912 for chance, where on v0 M9's was
*worse* than chance (4.050). And the mistakes stayed musical — mistake affinity 0.398
against a 0.264 chance baseline, MEA 0.508 against 0.275, both the best in the project.
The near-misses read like a musician's: Bageshree → Bheempalasi (0.74), Kafi → Bageshree
(0.73), Bihag → Hameer (0.72).

## What is left

**The tonic is still the lever, even annotated.** Two independent signals agree:

1. Oracle-rotation headroom is still **+0.077** over the annotated tonic.
2. **15.5 % of M9+'s test errors are near-exact rotations** of the true scale (chance 10.2 %)
   — Shivranjani → Hindol scores affinity 0.16 but **1.00 after rotating by 3 semitones**.

With a hand-annotated Sa those should not exist. They point at annotation errors on specific
videos, or at genuine within-recording drift the per-video constant cannot follow. The next
move is not a new method: it is **re-checking the annotation on the videos whose errors are
pure rotations**, which the confusion output now names directly.

Second: M9+'s remaining `w_tdms` was tuned to 1.5 and the sweep did not turn over — the
surface may deserve still more weight, and its resolution knobs (`n_bins`, `tau`) were only
coarsely searched at the new clip length.

## The metric was the bug  *(chi-square vs cosine)*

Every distribution-comparing method here scored with **cosine**, unexamined, from M9 onward.
Swapping in a symmetric chi-square distance — each bin weighed against its own magnitude
rather than contributing to an inner product — is worth more than any modelling change since
M3:

| method | cosine | **chi²** | gain |
|---|---|---|---|
| M9 melody surface | 0.341 | **0.402** | +0.061 |
| M11 pitch histogram | 0.341 | 0.382 | +0.041 |
| M9 re-tuned (n_bins 80) | — | **0.422** | |
| M9+ re-tuned (w_tdms 3.0) | 0.374 | **0.443** | +0.069 |

Cosine lets a handful of high-energy bins dominate the inner product, so a raag's long nyas
swars swamp the quiet-but-characteristic ones. Chi-square normalises each bin by its own
magnitude, and those quiet bins are exactly where two raags of the same scale differ.

The M9+ optimum is a genuine plateau, not a grid edge — `w_tdms` 2/3/4/6 all score 0.443,
falling to 0.422 (M9 alone) by w=20.

> **A comparison I got wrong, and the correction.** M11 was first run with chi² against an
> M9 that only ever had cosine, which made a plain histogram look equal to the 2-D surface
> and prompted the conclusion that *the delay axis buys nothing*. At a matched metric that
> is false: cosine 0.341 vs 0.341 (the delay axis really does add zero there), chi² 0.402 vs
> 0.382 (it adds +0.020). The lesson is narrower than "the delay axis is useless" — it is
> that **the metric mattered more than the representation**, for both.

## M11 — a pitch histogram and nothing else  *(the floor, and it is high)*

Octave-folded pitch histogram, duration-weighted, no motifs, no grammar, no database beyond
the candidate list. Per-raag templates are the mean of the training clips, refit per fold.
This is Chordia & Rae's 2007 pitch-class-distribution baseline with the bins left
un-rounded — and it is also **exactly M9's surface with the delay axis collapsed**, so the
pair measures what melodic *motion* adds over pitch *occupancy*.

| n_bins | metric | train CV |
|---|---|---|
| 12 (semitone) | cosine | 0.075 |
| 60 | cosine | 0.329 |
| 120 | cosine | 0.341 |
| **120** | **chi²** | **0.382** |

Two things this pins down:

1. **Quantizing to 12 bins costs 4.5x** (0.341 → 0.075). The strongest evidence yet for the
   un-quantized thesis, and only measurable now: bin positions are meaningless under a tonic
   that is wrong 38 % of the time.
2. **A method with no musical knowledge at all gets within 0.06 of the best one.** Fusing it
   with the phrase grammar makes it *worse* (0.382 → 0.364). Whatever the mukhyanga database
   is contributing, it is not contributing much through the M1-M7 route.

## Source separation  (`../source-separation`)

Split into its own reusable module — see [`../source-separation/plan.md`](../source-separation/plan.md)
for backends, the inspection CLI, and the audio-level findings. `extract.py --separate
<backend>` runs it before tracking and caches to
`cache/notes_<tracker>_<version>_<backend>.npz`.

Short version: **HPSS (free, no model) beats HT-Demucs** on this repertoire — it raises
CREPE's voiced fraction, cuts frame-to-frame jitter roughly tenfold, and makes the pitch
histogram peakier, while Demucs leaves jitter unchanged and actively removes melodic content
from instrumental clips (its `vocals` stem is the wrong target for a sitar or sarangi). The
jitter collapse was checked against drone-locking — a tracker sitting on the tanpura would
show the same low jitter — via the share of frames within 25 cents of Sa, which stayed flat.

## M12/M13/M14 — the database as a *prior*, not a model

M11 showed a knowledge-free histogram gets within 0.06 of the best method, which raises the
question of what the mukhyanga database is for. The answer is not "nothing" — it is that the
DB works as a **prior on a learned model**, and badly as a model on its own.

**M12 — histogram with a DB prior.** The template is built from *phrase occupancy*: how often
each swar appears across the raag's mukhyanga, aaroha and avaroha, spread onto the same
continuous bin grid M11 uses, then blended `ref = (1-λ)·learned + λ·db`.

**M13 — bigram LM with a DB prior.** Same idea over transitions: per-raag mean bigram matrix
from the training clips, blended toward the DB's phrase-derived transition model (which at
λ=1 is M3's model).

| λ (DB weight) | M12 histogram | M13 bigram LM |
|---|---|---|
| 0.0 — learned only | 0.382 | 0.275 |
| **0.3 — blended** | **0.405** | **0.304** |
| 0.6 | 0.370 | 0.301 |
| 1.0 — DB only, no training data | 0.217 | 0.237 |

The same shape twice: the blend beats both ends, and pure-DB is the worst of the three. The
database is a good prior and a poor model.

**Why phrase occupancy and not vaadi/samvaadi** — this is the post-mortem on M10:

    Bageshree    vaadi=m  samvaadi=S   scale = S R g m P D n
    Bheempalasi  vaadi=m  samvaadi=S   scale = S R g m P D n

Identical on every field M10 used. But their phrase unigrams differ at L1 0.43, and in the
musically right direction — Bageshree weakens P and leans on D, Bheempalasi leans on P. The
DB knows these raags apart; M10 was reading the wrong field.

**M14 — occupancy + transitions, both DB-guided.** M13 alone is much weaker than M12, but the
two are complementary:

| | train CV top-1 | top-5 | MRR |
|---|---|---|---|
| M13 bigram LM alone | 0.304 | — | — |
| M12 DB-histogram alone | 0.408 | — | — |
| **M14 = M12 + M13 (w=3)** | **0.464** | **0.787** | **0.606** |

Interior optimum, checked over w = 2…15 (0.462 / 0.464 / 0.455 / 0.454 / 0.442 / 0.428).
**+0.056 over the better half alone** — so *where* a raag sits and *how* it moves really are
different evidence, which is the first solid vindication of the phrase-structure idea since
the tonic landed. Best result in the project: **0.464 CV, ~23x chance.**

### What the M14 confusion matrix says  (`results/v1/confusion_m14_cv.png`)

Over 1810 train clips under grouped CV (top-1 0.466, top-5 0.785):

* **Solved:** the pentatonic and distinctive-scale raags — Malkauns 37/50, Madhukauns 31/45,
  Bairagi 30/40, Chandrakauns 27/40, Todi 21/30, Kalawati 12/15.
* **Barely recalled:** Keerwani 1/20, TilakKamod 3/20, PuriyaKalyan 4/35, Khamaj 5/35,
  Des 8/40, PuriyaDhanashri 8/35 — the seven-note Bilawal/Khamaj-thaat raags whose scales
  overlap heavily, exactly where pitch occupancy has least to say.
* **Hubs:** six labels absorb far more predictions than they own — Jaijaivanti 69 for 45 true
  clips, Des 69 for 40, Bihag 57, TilakKamod 53, Sohani 51, MaruBihag 50, against a 36-clip
  average. Des and TilakKamod are simultaneously over-predicted *and* badly recalled, the
  signature of a template that sits close to everything.

> **Hubness calibration was the obvious fix, and it failed.** Standardising each raag's score
> distribution over train — the trick that made M7 work — *costs* 3.5 points here: 0.464 →
> 0.429 (zscore), 0.418 (mean). M7's raw scores were badly-scaled log-likelihoods that needed
> it; chi² scores are already well-behaved, and z-scoring per raag also strips a genuine class
> prior (train is unbalanced, Malkauns n=50 against Kalawati n=15). The hubs are real and
> visible, but the cure has to be something other than rescaling.

### Where Bageshree vs Bheempalasi actually stands

The expectation was that DB guidance would separate this pair. Under grouped CV they were
**never the problem** — Bageshree→Bheempalasi is 0/40 for M12, 0/40 for M3, 1/40 for M11.
Bheempalasi instead fails *outward*: 9/40 correct, with the misses going to Malhar (5),
Bahar (4) and Pilu (4) — Kafi-thaat neighbours at scale Jaccard 0.70-0.88, all outside its
scale-twin group. Those errors cluster by recording (8 videos contributing 3-5 wrong clips
each), so whole performances fail together, which points at recording-level causes rather
than a scale-twin discrimination gap.

The DB prior does help twins in aggregate — twin-restricted top-1 goes 0.336 (M11) → 0.378
(M12), matching M9's 0.379 — just not through the pair that motivated it.

## Does source separation help?  No.  (`separation_effect.py`)

`../source-separation` makes the *pitch track* clearly better — CREPE's voiced fraction up,
frame-to-frame jitter down roughly tenfold, the histogram peakier. That is a claim about the
tracker. Whether it helps *classification* is a separate question, and the answer is no.

CREPE re-extracted over HPSS-separated audio (110 min for 1960 clips), everything else held
fixed. Train CV:

| method | unseparated | HPSS | delta |
|---|---|---|---|
| M9 melody surface | 0.422 | 0.393 | **-0.028** |
| M11 histogram | 0.382 | 0.380 | -0.001 |
| M12 histogram + DB | 0.405 | 0.393 | -0.012 |
| M13 bigram LM *(control)* | 0.304 | 0.304 | **+0.000** |
| M14 M12+M13 | 0.464 | 0.438 | -0.026 |

**M13 is the control** — it reads Tony's notes, which were not re-extracted, so its row must
not move. It doesn't, to three decimals, which is what says the other rows are measuring the
separation rather than a plumbing accident.

Every method that *does* see the separated audio gets slightly **worse**. The most likely
reason is the same fact that made the tracker metrics look so good: HPSS drops ~40 % of the
signal's energy and cuts jitter from ~18 cents to ~1. Some of that 18 cents was tabla bleed,
but a lot of it was **meend and gamak** — the continuous ornamentation between swars. A
120-bin histogram and a time-delayed melody surface are built specifically to see
sub-semitone movement, so smoothing it away costs them more than the tabla ever did.

Put plainly: **what looks like noise to a pitch tracker is partly the signal for raag
identification.** The tracker-level metrics in `../source-separation/plan.md` are real; they
just do not translate. Worth revisiting only with a separator that removes percussion without
smoothing the melodic line — which is the argument for a properly fine-tuned
BS-RoFormer over median filtering, not for tuning HPSS harder.
