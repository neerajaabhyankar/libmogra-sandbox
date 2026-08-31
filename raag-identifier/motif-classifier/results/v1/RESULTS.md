# Results

50-way raag identification on `hindustani-raag-small` (v1). Chance = 0.020
(1/50). Train numbers are 5-fold **grouped-by-video** CV on the
train split (used for tuning); test numbers are a single pass over the held-out test
split with the config that CV chose, run once.

| method | train top-1 (CV) | **test top-1** | test top-5 | test MRR | test top-1 by video-vote | confusion matrix |
|---|---|---|---|---|---|---|
| **m3** — phrase grammar, smoothed bigram log-likelihood | 0.285 ± 0.005 | **0.227** (34/150) | 0.607 | 0.385 | 0.420 | [`confusion_m3.png`](confusion_m3.png) |
| **m4** — M3 grammar over Tony **+ CREPE** fused | 0.291 ± 0.004 | **0.253** (38/150) | 0.580 | 0.416 | 0.420 | [`confusion_m4.png`](confusion_m4.png) |
| **m7** — channel + CREPE + tonic prior + per-raag calibration | 0.332 ± 0.002 | **0.273** (41/150) | 0.607 | 0.437 | 0.500 | [`confusion_m7.png`](confusion_m7.png) |
| **m9** — time-delayed melody surface — un-quantized contour, **no** mukhyanga | 0.414 ± 0.006 | **0.360** (54/150) | 0.653 | 0.509 | 0.480 | [`confusion_m9.png`](confusion_m9.png) |
| **m9plus** — M4 + melody surface (quantized phrases + continuous contour) | 0.438 ± 0.004 | **0.367** (55/150) | 0.693 | 0.522 | 0.540 | [`confusion_m9plus.png`](confusion_m9plus.png) |
| **m11** — pitch histogram only — no motifs, no grammar, no database | 0.385 ± 0.004 | **0.360** (54/150) | 0.647 | 0.498 | 0.440 | [`confusion_m11.png`](confusion_m11.png) |
| **m12** — pitch histogram with the mukhyanga phrase inventory as a prior | 0.407 ± 0.003 | **0.373** (56/150) | 0.653 | 0.512 | 0.480 | [`confusion_m12.png`](confusion_m12.png) |
| **m13** — bigram transition LM, learned, with the DB as a prior | 0.315 ± 0.007 | **0.287** (43/150) | 0.587 | 0.431 | 0.380 | [`confusion_m13.png`](confusion_m13.png) |
| **m14** — **M12 + M13** — occupancy and transitions, both DB-guided | 0.462 ± 0.001 | **0.360** (54/150) | 0.707 | 0.521 | 0.480 | [`confusion_m14.png`](confusion_m14.png) |
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
| **m3** | 0.358 | 0.263 | 0.402 | 0.275 | 3.502 | 3.103 | 0.078 | 0.105 | 0.716 | 0.685 |
| **m4** | 0.386 | 0.259 | 0.424 | 0.275 | 3.460 | 2.979 | 0.125 | 0.099 | 0.721 | 0.682 |
| **m7** | 0.367 | 0.267 | 0.447 | 0.275 | 3.384 | 2.766 | 0.046 | 0.107 | 0.711 | 0.688 |
| **m9** | 0.398 | 0.264 | 0.549 | 0.275 | 4.268 | 3.020 | 0.156 | 0.102 | 0.749 | 0.684 |
| **m9plus** | 0.416 | 0.261 | 0.558 | 0.275 | 3.999 | 2.749 | 0.168 | 0.098 | 0.750 | 0.681 |
| **m11** | 0.417 | 0.264 | 0.517 | 0.275 | 4.265 | 3.317 | 0.146 | 0.094 | 0.756 | 0.680 |
| **m12** | 0.422 | 0.263 | 0.522 | 0.275 | 3.982 | 3.040 | 0.160 | 0.102 | 0.768 | 0.682 |
| **m13** | 0.384 | 0.261 | 0.481 | 0.275 | 4.087 | 3.230 | 0.084 | 0.102 | 0.731 | 0.682 |
| **m14** | 0.441 | 0.265 | 0.573 | 0.275 | 4.101 | 2.864 | 0.156 | 0.103 | 0.777 | 0.684 |

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

**m7**
```
representation: {'tracker': 'tony', 'note_source': 'hmm', 'tonic_mode': 'true', 'tonic_refine': True, 'min_dur': 0.0, 'max_cents_dev': 2400.0, 'collapse_repeats': True}
method:         {'use_channel': True, 'channel_kw': {'p_self': 0.2, 'prior': 5.0, 'emission_temp': 0.3, 'w_dur': 1.0}, 'base_kw': {'lam_bi': 0.7, 'lam_uni': 0.1, 'uni_from_scale': 0.75}, 'w_crepe': 1.0, 'calibrate': 'zscore', 'marginalise_tonic': False, 'temperature': 0.1}
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

## Error structure (m12, test)

- correct at rank 1: AheerBhairav, Bageshree, Bairagi, Bhairav, Bhairavi, Bhoopali, Bihag, Charukeshi, DarbariKanada, Deshkar, Dhani, Durga, Hameer, HansDhwani, Hindol, Jaijaivanti, Jog, Kafi, Lalit, Madhukauns, Madhuvanti, Malhar, Malkauns, MaruBihag, Marwa, Multani, PuriyaDhanashri, Sarang, Shankara, Shree, TilakKamod, Vibhas, Yaman
- median rank of the true raag: 2 of 50
- most-predicted labels: Hameer (10), PuriyaDhanashri (8), Sarang (8), Bahar (7), Jaijaivanti (7), Bheempalasi (7)

Most defensible misses (highest affinity):

  - PuriyaDhanashri → Basant  (affinity 0.75)
  - Basant → PuriyaDhanashri  (affinity 0.75)
  - Basant → PuriyaDhanashri  (affinity 0.75)
  - Basant → PuriyaDhanashri  (affinity 0.75)
  - Bageshree → Bheempalasi  (affinity 0.74)
  - Kafi → Bheempalasi  (affinity 0.74)

Least defensible misses (lowest affinity); `rot` is the affinity after rotating the
predicted scale by `k` semitones — a high `rot` means this was a tonic error:

  - Shivranjani → Lalit  (affinity 0.04, rot 0.50 at k=1)
  - Shivranjani → Marwa  (affinity 0.12, rot 0.83 at k=3)
  - AlhaiyaBilawal → Madhukauns  (affinity 0.14, rot 0.62 at k=4)
  - Shivranjani → Hindol  (affinity 0.16, rot 1.00 at k=3)
  - Chandrakauns → TilakKamod  (affinity 0.16, rot 0.50 at k=1)
  - Keerwani → KaushikDhwani  (affinity 0.17, rot 0.71 at k=3)
