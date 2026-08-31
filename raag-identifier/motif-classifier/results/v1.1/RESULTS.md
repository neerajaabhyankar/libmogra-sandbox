# Results

50-way raag identification on `hindustani-raag-small` (v1.1). Chance = 0.020
(1/50). Train numbers are 5-fold **grouped-by-video** CV on the
train split (used for tuning); test numbers are a single pass over the held-out test
split with the config that CV chose, run once.

| method | train top-1 (CV) | **test top-1** | test top-5 | test MRR | test top-1 by video-vote | confusion matrix |
|---|---|---|---|---|---|---|
| **m3** — phrase grammar, smoothed bigram log-likelihood | 0.290 ± 0.005 | **0.240** (36/150) | 0.640 | 0.404 | 0.440 | [`confusion_m3.png`](confusion_m3.png) |
| **m4** — M3 grammar over Tony **+ CREPE** fused | 0.295 ± 0.004 | **0.267** (40/150) | 0.620 | 0.436 | 0.440 | [`confusion_m4.png`](confusion_m4.png) |
| **m9** — time-delayed melody surface — un-quantized contour, **no** mukhyanga | 0.422 ± 0.006 | **0.393** (59/150) | 0.687 | 0.538 | 0.500 | [`confusion_m9.png`](confusion_m9.png) |
| **m9plus** — M4 + melody surface (quantized phrases + continuous contour) | 0.447 ± 0.004 | **0.400** (60/150) | 0.720 | 0.551 | 0.560 | [`confusion_m9plus.png`](confusion_m9plus.png) |
| **m11** — pitch histogram only — no motifs, no grammar, no database | 0.398 ± 0.004 | **0.387** (58/150) | 0.673 | 0.526 | 0.480 | [`confusion_m11.png`](confusion_m11.png) |
| **m12** — pitch histogram with the mukhyanga phrase inventory as a prior | 0.414 ± 0.004 | **0.400** (60/150) | 0.673 | 0.539 | 0.500 | [`confusion_m12.png`](confusion_m12.png) |
| **m13** — bigram transition LM, learned, with the DB as a prior | 0.323 ± 0.007 | **0.320** (48/150) | 0.627 | 0.462 | 0.400 | [`confusion_m13.png`](confusion_m13.png) |
| **m14** — **M12 + M13** — occupancy and transitions, both DB-guided | 0.468 ± 0.002 | **0.373** (56/150) | 0.733 | 0.543 | 0.500 | [`confusion_m14.png`](confusion_m14.png) |
| _chance_ | 0.020 | 0.020 | 0.100 | 0.090 | 0.020 | — |

## How bad are the mistakes?

Accuracy scores Tilak Kamod → Des (a near-miss any listener could make) exactly as
badly as Tilak Kamod → Bairagi (nothing in common). These grade against
`raagspace.affinity()`, a raag-to-raag similarity built only from the libmogra
database — TF-IDF over mukhyanga/aaroha n-grams, swar-set Jaccard, and thaat.
Each metric sits next to the value random guessing would score.

- **mistake affinity** — mean affinity(true, predicted) over errors only. Higher = misses land nearby.
- **expected affinity (MEA)** — `Σ_r p(r)·affinity(true,r)` over the whole softmax output, not just the argmax.
- **affinity CE** — cross-entropy against a soft target `q ∝ affinity(true,·)^4`. **Lower is better.** This is the mukhyanga-based loss: it does not punish mass on genuinely related raags.
- **NLL** — ordinary negative log-likelihood of the true raag, for reference (chance = ln 50 = 3.912).
- **tonic-explained** — of the errors, the share whose prediction is a near-exact *rotation* of the true scale. Those are Sa-placement failures, not raag failures.

Softmax temperature is calibrated per method on **train** (standard temperature
scaling, minimising NLL) so methods with different score scales are comparable.

| method | mistake affinity | (chance) | MEA | (chance) | affinity CE ↓ | NLL ↓ | tonic-explained | (chance) | rot-affinity | (chance) |
|---|---|---|---|---|---|---|---|---|---|---|
| **m3** | 0.372 | 0.263 | 0.423 | 0.275 | 3.377 | 2.890 | 0.044 | 0.106 | 0.713 | 0.685 |
| **m4** | 0.404 | 0.259 | 0.447 | 0.275 | 3.291 | 2.695 | 0.091 | 0.101 | 0.721 | 0.682 |
| **m9** | 0.411 | 0.265 | 0.573 | 0.275 | 3.949 | 2.566 | 0.132 | 0.104 | 0.750 | 0.684 |
| **m9plus** | 0.436 | 0.264 | 0.598 | 0.275 | 3.869 | 2.311 | 0.133 | 0.100 | 0.755 | 0.682 |
| **m11** | 0.431 | 0.266 | 0.536 | 0.275 | 3.906 | 2.855 | 0.120 | 0.096 | 0.760 | 0.681 |
| **m12** | 0.445 | 0.267 | 0.540 | 0.275 | 3.642 | 2.601 | 0.133 | 0.105 | 0.779 | 0.685 |
| **m13** | 0.392 | 0.260 | 0.497 | 0.275 | 3.876 | 2.964 | 0.049 | 0.101 | 0.718 | 0.682 |
| **m14** | 0.460 | 0.265 | 0.606 | 0.275 | 3.971 | 2.458 | 0.117 | 0.104 | 0.776 | 0.685 |

## Chosen configurations

**m3**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'shift_mode': 'none', 'w_arohana': 1.0, 'symmetric': False, 'lam_bi': 0.7, 'lam_uni': 0.1, 'uni_from_scale': 0.75, 'nyas_boost': 1.0, 'w_dur': 1.0, 'dur_weighted': False, 'w_skip': 0.5}
features:       {}
```

**m4**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'w_crepe': 0.6, 'primary': 'tony'}
features:       {}
```

**m9**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'tracker': 'crepe', 'n_bins': 80, 'tau': 0.3, 'smooth': 1.0, 'tonic_mode': 'true', 'metric': 'chi2'}
features:       {}
```

**m9plus**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'w_tdms': 3.0, 'base': 'm4', 'base_kw': {'w_crepe': 1.0, 'primary': 'tony'}, 'tdms_kw': {'tracker': 'crepe', 'n_bins': 80, 'tau': 0.3, 'tonic_mode': 'true', 'metric': 'chi2'}}
features:       {}
```

**m11**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'n_bins': 240, 'source': 'frames', 'tracker': 'crepe', 'metric': 'chi2', 'smooth': 2.0, 'power': 0.5, 'tonic_mode': 'true', 'separate': None}
features:       {}
```

**m12**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'n_bins': 120, 'source': 'frames', 'tracker': 'crepe', 'metric': 'chi2', 'smooth': 1.0, 'power': 0.5, 'tonic_mode': 'true', 'separate': None, 'lam': 0.15}
features:       {}
```

**m13**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'lam_db': 0.5, 'which': 'bigram_dur', 'w_uni': 0.3, 'order': 2}
features:       {}
```

**m14**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'w_tdms': 3.0, 'base': 'm13', 'base_kw': {'lam_db': 0.3, 'which': 'bigram_dur', 'w_uni': 0.3}, 'tdms_kw': {'n_bins': 120, 'source': 'frames', 'tracker': 'crepe', 'metric': 'chi2', 'smooth': 1.0, 'power': 0.5, 'tonic_mode': 'true', 'separate': None, 'lam': 0.3}, 'tdms_cls': 'm12'}
features:       {}
```

## Error structure (m9plus, test)

- correct at rank 1: AheerBhairav, Bahar, Bairagi, Bhairav, Bhairavi, Bhoopali, Bihag, Chandrakauns, Charukeshi, DarbariKanada, Deshkar, Dhani, Durga, Hameer, HansDhwani, Hindol, Jaijaivanti, Jog, Kafi, Lalit, Madhukauns, Madhuvanti, Malhar, Malkauns, MaruBihag, Marwa, Sarang, Shankara, Shivranjani, Shree, TilakKamod, Todi, Vibhas, Yaman
- median rank of the true raag: 2 of 50
- most-predicted labels: MaruBihag (10), Jaijaivanti (9), AlhaiyaBilawal (9), Sarang (9), PuriyaDhanashri (6), Bihag (6)

Most defensible misses (highest affinity):

  - PuriyaDhanashri → Basant  (affinity 0.75)
  - Basant → PuriyaDhanashri  (affinity 0.75)
  - Basant → PuriyaDhanashri  (affinity 0.75)
  - Basant → PuriyaDhanashri  (affinity 0.75)
  - Kafi → Bageshree  (affinity 0.73)
  - MaruBihag → Bihag  (affinity 0.70)

Least defensible misses (lowest affinity); `rot` is the affinity after rotating the
predicted scale by `k` semitones — a high `rot` means this was a tonic error:

  - Vibhas → MaruBihag  (affinity 0.15, rot 0.44 at k=1)
  - Vibhas → MaruBihag  (affinity 0.15, rot 0.44 at k=1)
  - Keerwani → KaushikDhwani  (affinity 0.17, rot 0.71 at k=3)
  - Bairagi → Todi  (affinity 0.19, rot 0.50 at k=4)
  - Madhuvanti → PuriyaDhanashri  (affinity 0.21, rot 0.56 at k=3)
  - Tilang → Shree  (affinity 0.21, rot 0.62 at k=4)
