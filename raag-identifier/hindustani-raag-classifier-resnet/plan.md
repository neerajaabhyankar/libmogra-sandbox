# Hindustani Raag Classifier — ResNet (jeevster backbone) — Plan

## Background / groundwork (from `embeddings-exploration/`)

- `embeddings-exploration/models/crc_jeevster.py` wraps **`carnatic-raga-classifier-jeevster`**
  (symlinked, read-only) — a pretrained 1D-ResNet raga classifier:
  `Conv1d stem (2→300ch, k=80, stride=16)` → `10 × ResidualBlock(300ch, k=3, +MaxPool1d(2))`
  → global avg-pool over time → `fc1: Linear(300→150)` → log_softmax.
  Checkpoint: `carnatic-raga-classifier-jeevster/ckpts/best_ckpt.tar` (epoch 53, ~7M params,
  ~66 MB), trained on 30s @ 8kHz stereo, per-channel-normalized Carnatic audio (150 ragas).
- We extracted the **300-dim `fc1`-input vector** (global-avg-pooled, pre-classification-head)
  as a clip-level embedding for our Hindustani dataset (`neerajaabhyankar/hindustani-raag-small`,
  rev `0dfb021e54e0e7489b90a47e23ef15f34fa740ec`), `whole_clip=True` (no chunking — see
  `embeddings-exploration/crc_jeevster.md` for the input-length constraints and rationale).
- Saved to `embeddings-exploration/outputs/2s/crc-jeevster/{train,test}_<idx>.npz`, each with:
  - `chunks`: `(1, 300)` — single whole-clip embedding
  - `clip_mean`: `(300,)` — same vector (mean over 1 chunk)
  - `clip_rich`: `(900,)` — **degenerate** for this model: `[mean, std=0, max=mean]`, so just
    `[emb, zeros(300), emb]`. Not useful — ignore `clip_rich`, use `clip_mean` only.
- Coverage: only `LABEL_INDICES = range(0, 5)` so far, i.e. 5 of the dataset's 50 raag classes:
  `AheerBhairav(0), AlhaiyaBilawal(1), Bageshree(2), Bahar(3), Bairagi(4)`.
  - **train**: 123 clips — counts `{0: 33, 1: 30, 2: 24, 3: 12, 4: 24}`
  - **test**: 8 clips — counts `{0: 2, 2: 2, 3: 2, 4: 2}` — **class 1 entirely missing**, far
    too small/unbalanced to be a meaningful held-out test set on its own.
- Conclusion from Attempt 3 (`plan.md`): no visible raag separation in 2D UMAP of `clip_mean` —
  but UMAP-on-25-points/class is a weak test. The real test is **supervised**: can a small
  classifier head actually learn raag from these 300-dim embeddings?

## Goal

Build `RaagResNetClassifier` = jeevster's pretrained ResNet backbone (conv stem + 10 residual
blocks + global avg-pool, **without** the original 150-way `fc1`) + a new small MLP head trained
on Hindustani raag labels. Package it as a clean, HF-pushable module.

---

## Stage 0 — Sanity probe on precomputed embeddings (no audio, no backbone yet)

**Goal:** with zero new feature extraction, check whether `clip_mean` (300-dim) embeddings for
the 123 train clips (5 classes) are separable at all by a tiny classifier.

- Load all `outputs/2s/crc-jeevster/train_*.npz` → `clip_mean` vectors + labels (via the HF
  dataset's `label` field, matched by the `<idx>` in the filename — same approach used for the
  Counter check above).
- Dataset is tiny (123 samples / 5 classes, imbalanced 33/30/24/12/24) and the provided `test`
  split is too small/unbalanced (8 samples, missing class 1) to be a standalone held-out set.
  → Use **stratified k-fold cross-validation** (k=5) over the 123 train clips as the primary
  evaluation. The 8 official test clips can be reported separately as an extra (very rough)
  sanity check, not as the headline metric.
- Model: simplest possible MLP head — `Linear(300 → hidden) → ReLU → Linear(hidden → 5)`
  (or even a pure linear probe `Linear(300 → 5)` as a baseline). Standardize features
  (zero mean / unit variance, fit on train fold only) before feeding in.
- Train with cross-entropy, Adam, early stopping on val-fold loss (dataset is tiny — a few
  hundred epochs at most, should run in seconds on CPU).
- **Output:** per-fold accuracy + macro-F1 (macro-F1 matters given class imbalance), and an
  **aggregated 5×5 confusion matrix** (sum of all fold val-set confusions), saved as a PNG
  under `hindustani-raag-classifier-resnet/outputs/probe/`.
- **Decision point:** if the linear probe / 1-hidden-layer MLP is clearly above chance
  (chance ≈ 1/5 = 20%, but with class imbalance the majority-class baseline ≈ 27%), proceed to
  Stage 1. If it's at chance, that's important to know before investing in the full pipeline —
  flag it and discuss before continuing.

**Script:** `01_probe_embeddings.py` — self-contained, reads only the `.npz` files + dataset
labels, no audio loading, no backbone.

---

## Stage 1 — Hyperparameter exploration on the embedding probe

Using the same CV harness from Stage 0, sweep over MLP head configs:

| Knob | Values to try |
|---|---|
| Depth | linear probe (0 hidden layers), 1 hidden layer, 2 hidden layers |
| Width | 64, 128, 256 |
| BatchNorm | on / off |
| Dropout | 0.0, 0.2, 0.5 |
| Weight decay | 0.0, 1e-4, 1e-2 |
| Learning rate | 1e-3, 3e-4 |

- Given how small the dataset is, expect **dropout + weight decay to matter a lot** (high
  overfitting risk with only ~100 training points/fold and 300-dim inputs) and **2 hidden
  layers to likely overfit** unless heavily regularized.
- Rank configs by **mean CV macro-F1**, not raw accuracy (class imbalance).
- Output: a small results table (config → mean/std accuracy & macro-F1 across folds) + the
  confusion matrix for the best config. Save table as CSV/markdown under `outputs/probe/`.
- This stage informs the head architecture (depth/width/batchnorm/dropout) used in Stage 2 —
  no need to re-derive it later.

**Script:** extends `01_probe_embeddings.py` (or a sibling `02_sweep_head.py`) — still
embedding-only, fast iteration.

---

## Stage 2 — Clean, HF-pushable module

Once Stage 1 picks a head architecture, build the real package: backbone (from jeevster's
checkpoint) + chosen head, operating on **raw audio** end-to-end (so it's a real deployable
model, not just an embedding-probe script).

Proposed layout:

```
hindustani-raag-classifier-resnet/
  plan.md
  outputs/
    probe/                      # Stage 0/1 results (confusion matrices, sweep tables)
  raag_resnet/
    __init__.py
    configuration.py            # RaagResNetConfig (HF PretrainedConfig subclass):
                                 #   backbone hyperparams (n_channel=300, n_blocks=10, stride=16,
                                 #   max_pool_every=1) + head hyperparams (depth/width/batchnorm/
                                 #   dropout from Stage 1) + num_labels + id2label/label2id
    modeling.py                 # RaagResNetForAudioClassification (HF PreTrainedModel subclass):
                                 #   - backbone: conv_first + res_blocks, ported from jeevster's
                                 #     models.ResNetRagaClassifier, weights loaded from
                                 #     best_ckpt.tar (state_dict, fc1 excluded)
                                 #   - head: new MLP (Stage 1's chosen config) -> num_labels
                                 #   - forward(input_values) -> logits; supports
                                 #     freeze_backbone flag
    feature_extraction.py       # RaagResNetFeatureExtractor (HF FeatureExtractor subclass):
                                 #   resample->8kHz, mono->stereo, per-channel normalize,
                                 #   zero-pad to 5s floor (40000 samples) -- ports the
                                 #   preprocessing from embed.py's CRCJeevsterEmbedder
  train.py                      # end-to-end training on neerajaabhyankar/hindustani-raag-small
                                 #   (labels 0-4 to start), backbone frozen by default
  evaluate.py                   # confusion matrix + metrics on held-out fold/test split
  push_to_hub.py                # pushes model + config + feature extractor to HF Hub
```

Key implementation notes:
- **Loading the backbone:** reuse the `importlib`-based loading trick from
  `embeddings-exploration/models/crc_jeevster.py` to instantiate jeevster's
  `ResNetRagaClassifier`, load `best_ckpt.tar`, then copy `conv_first` + `res_blocks` weights
  into our own module (don't keep a dependency on the symlinked jeevster folder at runtime —
  the new module should be self-contained for HF push).
- **Backbone frozen vs fine-tuned:** default to **frozen backbone** (matches the embedding-probe
  setup validated in Stage 0/1, and 123 training clips is too few to safely fine-tune a 7M-param
  ResNet). Make it a config flag (`freeze_backbone: bool`) so we can experiment with unfreezing
  the last residual block(s) later if frozen-backbone accuracy plateaus.
- **Preprocessing parity:** `feature_extraction.py` must exactly match `CRCJeevsterEmbedder.embed`
  (resample to 8kHz, mono→stereo via `repeat`, pad to 40000 samples, per-channel normalize) so
  that end-to-end model output matches the Stage 0/1 probe results when the head is identical.
- **Class coverage:** Stage 0/1 only cover 5 classes (embeddings already computed). Before
  pushing, decide whether to (a) ship a 5-class model matching what we validated, or (b) compute
  embeddings for more/all 50 classes first (would require re-running
  `embeddings-exploration/main.py embed` with expanded `LABEL_INDICES` — out of scope for this
  plan, flagged as a follow-up).

---

## Open decisions (for discussion before/while executing)

1. **Backbone freeze vs. fine-tune** — default frozen; revisit if accuracy is poor.
2. **Class scope for the pushed model** — 5 classes (current embeddings) vs. expanding to more
   raags first (requires new embedding extraction, separate effort).
3. **HF repo name/visibility** for the eventual push (e.g.
   `neerajaabhyankar/hindustani-raag-classifier-resnet`).
4. **Eval protocol for the final module** — stratified k-fold (consistent with Stage 0/1) vs.
   a fixed train/val split once we're working with raw audio + augmentation.

---

## Execution order

1. Stage 0 (probe, ~1 script, runs in seconds) → confusion matrix, go/no-go signal.
2. Stage 1 (sweep, extends Stage 0 script) → pick head architecture.
3. Stage 2 (clean module) → only after Stage 0/1 give a sensible head config and the user
   confirms scope (open decisions above).
4. Push to HF — only after the user is happy with Stage 2 results.

---

## Findings

### Stage 0 — sanity probe (`01_probe_embeddings.py`)

`Linear(300→128) → ReLU → Dropout(0.2) → Linear(128→5)`, Adam, wd=1e-2, 5-fold CV:

- **CV mean accuracy 0.659 ± 0.068, macro-F1 0.618 ± 0.085** — well above majority-class
  baseline (0.268) and chance (0.20). Clearly trainable.
- 8-clip official test set: 5/8 (0.625), consistent with CV.
- Confusion matrix: `outputs/probe/stage0_cv_confusion_matrix.png`. Strong diagonal for
  AheerBhairav (24/33), AlhaiyaBilawal (24/30), Bageshree (15/24), Bairagi (12/24); Bahar
  weakest (6/12 — also the smallest class, 12 samples).

**Decision:** clear go — proceeded to Stage 1.

### Stage 1 — head sweep (`02_sweep_head.py`)

222 configs (linear probe + 1/2-hidden-layer MLPs × width {64,128,256} × batchnorm × dropout
{0,0.2,0.5} × weight_decay {0,1e-4,1e-2} × lr {1e-3,3e-4}), same 5 CV folds for all configs.
Every config's final model (trained on all 123 clips for `median(best_epoch)+1` epochs) is
checkpointed to `outputs/sweep/checkpoints/cfg_NNN.pt` (state_dict + scaler + config + CV
metrics, 222 files, ~53 MB total). Full ranked table: `outputs/sweep/results.csv`.

**Best config** — `cfg_030`: depth=1, width=64, **batchnorm=True**, dropout=0.2, wd=0.0, lr=1e-3,
55 epochs:
- **CV mean accuracy 0.692 ± 0.079, macro-F1 0.671 ± 0.093** — modest but consistent
  improvement over Stage 0 (+0.033 acc, +0.053 macro-F1).
- Confusion matrix: `outputs/sweep/stage1_best_confusion_matrix.png`. Improved across the
  board vs. Stage 0, notably Bageshree (19/24 vs 15/24) and Bahar (7/12 vs 6/12).
- Top-10 configs are dominated by **batchnorm=True**, depth 1-2, width 64-128 — batchnorm
  on a 1-hidden-layer head is the clearest consistent win. Dropout 0.2-0.5 helps; weight
  decay has little effect once batchnorm+dropout are present.

**Decision for Stage 2 head architecture:** `Linear(300→64) → BatchNorm1d(64) → ReLU →
Dropout(0.2) → Linear(64→num_classes)`, ~55 training epochs as a starting point (re-tune if
class count/dataset size changes).

### Stage 2 — assembled model (`raag_resnet/`, `03_build_model.py`)

Built `RaagResNetForAudioClassification` (HF `PreTrainedModel`/`PretrainedConfig`/
`FeatureExtractionMixin`, package under `raag_resnet/`):

- **Backbone**: `conv_first` + `res_blocks` ported from jeevster's `ResNetRagaClassifier`
  (module names match exactly), weights loaded from `best_ckpt.tar` (`fc1` dropped).
  `freeze_backbone=True` (config flag) — backbone always runs under `torch.no_grad()` and
  stays in `eval()` even when the model is in `train()` mode (BatchNorm running stats
  preserved).
- **Head + scaler**: loaded directly from Stage 1's `outputs/sweep/checkpoints/cfg_030.pt`
  (head weights + `feat_mean`/`feat_scale` buffers = the StandardScaler fit in Stage 1) — no
  retraining, exact reuse of the validated Stage 1 model.
- **Feature extractor**: `RaagResNetFeatureExtractor`, ports `CRCJeevsterEmbedder.embed`'s
  preprocessing (resample→8kHz, mono→stereo, zero-pad to 40000 samples, per-channel
  normalize).

**Verification:**
- Parity check: `backbone_forward(feature_extractor(raw_audio))` vs precomputed
  `clip_mean` from `embeddings-exploration/outputs/2s/crc-jeevster/` — **exact match
  (0.00e+00 max abs diff)** on the first 5 train clips. The ported backbone is numerically
  identical to the original embedder.
- `save_pretrained` / `from_pretrained` round-trip verified — config, feature extractor, and
  backbone parity all preserved after reload.
- End-to-end eval (raw audio → logits, all on the assembled model):
  - **train (123 clips): 100% accuracy** — this is the cfg_030 model's fit on its own
    training data (it was trained on all 123 clips with no held-out val), so this is
    expected memorization, not a generalization estimate. The honest estimate remains
    Stage 1's CV macro-F1 (0.671 ± 0.093).
  - **test (8 clips): 50% (4/8)** — `outputs/model_eval/stage2_test_confusion_matrix.png`.
    Lower than Stage 0's reference run (5/8) on this same tiny 8-clip set; with n=8 this is
    within noise (1 clip = 12.5%).

**Saved artifacts** (local only, nothing pushed):
- `outputs/model/` — `config.json`, `model.safetensors` (~21 MB), `preprocessor_config.json`.
  Loadable via `RaagResNetForAudioClassification.from_pretrained("outputs/model")` and
  `RaagResNetFeatureExtractor.from_pretrained("outputs/model")`.
- `outputs/model_eval/stage2_{train,test}_confusion_matrix.png`.

**Open items before any push (per original "Open decisions"):** still 5-class only (matches
Stage 0/1 embedding coverage); backbone still frozen; HF repo naming/visibility undecided.
Push deliberately not done in this stage.

### Stage 3 — all 50 classes (`04_compute_embeddings_50class.py`, `05_train_full_model.py`)

Expanded from the 5-class subset to the full dataset (1161 train clips, 92 test clips, 50
raag classes). Did **not** touch `embeddings-exploration/outputs` (only covers labels 0-4) or
any Stage 0-2 artifacts — new embeddings/model live entirely under
`outputs/embeddings_all/` and `outputs/full50/`.

- **Embeddings**: computed via the Stage 2 backbone (`backbone_forward` + feature extractor,
  proven numerically identical to crc-jeevster's embedder) for all 1253 clips (~1m23s).
  Saved to `outputs/embeddings_all/{train,test}.npz` (`X` (N,300), `y`, `idx`). Train class
  counts: 12-42 per class. Test: 2 per class for 46/50 classes, **0 test samples for classes
  1, 18, 27, 30** (AlhaiyaBilawal, Durga, Khamaj, Lalit) — those rows are necessarily empty in
  the test confusion matrix.
- **Split**: stratified 85/15 train_inner (986) / val (175) from the 1161 train clips
  (`StratifiedShuffleSplit`, seed=0).
- **Head**: same architecture as Stage 1's `cfg_030` (Linear(300→64)→BatchNorm1d→ReLU→
  Dropout(0.2)→Linear(64→50)), wd=0, lr=1e-3 — **not re-swept for 50 classes** (flagged as a
  follow-up below). Full-batch training, early stopping on val loss (patience=30).
- **Training curves**: `outputs/full50/training_curves.png`. Stopped at epoch 161, best
  epoch=130 (val_loss=3.127, val_acc=0.223, val_f1=0.178). Curves show clear overfitting:
  train macro-F1/acc climb to ~0.96 by epoch 160 while val plateaus around 0.15-0.23 from
  ~epoch 80 onward — the 64-unit bottleneck (and ~20 train samples/class) limits
  generalization well below train fit.
- **Test results** (92 clips, `outputs/full50/test_confusion_matrix.png`): **accuracy 0.109
  (10/92), macro-F1 0.102**. Chance = 1/50 = 0.02, so ~5x chance, but far below the 5-class
  Stage 1/2 numbers (~0.65-0.69) — expected, given 10x more classes with similar total data.

**Saved artifacts** (local only, nothing pushed):
- `outputs/embeddings_all/{train,test}.npz` — all 1253 backbone embeddings + labels.
- `outputs/full50/training_curves.png`, `outputs/full50/test_confusion_matrix.png`.
- `outputs/full50/head_checkpoint.pt` — head state_dict + scaler + config + val metrics
  (same format as Stage 1's `cfg_*.pt`).
- `outputs/full50/model/` — full assembled `RaagResNetForAudioClassification`
  (`config.json`, `model.safetensors`, `preprocessor_config.json`), `id2label`/`label2id`
  cover all 50 raags. Loadable the same way as Stage 2's `outputs/model/`.

**Follow-ups (not done):** re-sweep head hyperparams for 50 classes (wider head likely needed
given the 64-unit bottleneck; probably more dropout/weight decay given ~20 samples/class);
investigate the 4 zero-test-sample classes; backbone still frozen/5-class-trained-then-reused
(never fine-tuned on Hindustani audio at all — could be worth unfreezing later with this much
more data).

### Stage 4 — full fine-tune, all 50 classes (`06_finetune_full_model.py`)

Unfroze the entire backbone (10 residual blocks + conv stem) and trained it end-to-end
together with a freshly-initialized head, starting from the same `best_ckpt.tar` backbone
weights as Stage 2/3. Train split only (1161 clips); same 85/15 `StratifiedShuffleSplit`
(seed=0) → train_inner (986) / val (175), identical split to Stage 3 since it's the same
seed applied to the same ordered labels. Did not touch any Stage 0-3 outputs — everything new
lives under `outputs/full50_finetuned/`.

- **Why raw audio now**: with the backbone trainable, embeddings can no longer be precomputed
  once — every step needs a fresh forward+backward through the ResNet. To keep this tractable
  on CPU, training used random 5s crops (40000 samples @ 8kHz, batch_size=8) so BatchNorm sees
  batch_size>1; val used center crops of the same length (so BatchNorm running stats, which
  are now updated during training, are evaluated at the same input scale). Final test eval
  used full-length clips via `RaagResNetFeatureExtractor`, matching Stage 2/3's deployed
  preprocessing.
- **Head**: same architecture as Stage 3 (Linear(300→64)→BatchNorm1d→ReLU→Dropout(0.2)→
  Linear(64→50)) but freshly initialized — full fine-tuning gets to relearn the head from
  scratch jointly with the backbone, rather than warm-starting from Stage 3's head.
- **Optimizer**: discriminative LRs — backbone 1e-5 (wd=1e-4), head 1e-3 (wd=0), Adam,
  grad-norm clipped to 5.0. Up to 40 epochs, early stopping on val loss (patience=8).
- **Result — training was very slow and unstable, and stopped early (epoch 9, best
  epoch=1)**: train loss only dropped from 3.94→3.32 (train acc 0.04→0.17) over 10 epochs —
  the 1e-5 backbone LR makes the backbone barely move in this budget. Val loss/acc were
  extremely noisy (val loss spiked to ~7.0 at epoch 5, val acc bounced between 0 and 0.12)
  rather than smoothly tracking train, so early stopping (patience=8) latched onto epoch 1
  — essentially a lucky early checkpoint, not a converged model.
  See `outputs/full50_finetuned/training_curves.png`.
- **Test results** (`outputs/full50_finetuned/test_confusion_matrix.png`): **accuracy 0.054
  (5/92), macro-F1 0.019** — worse than Stage 3's frozen-backbone result (0.109/0.102), and
  the confusion matrix shows the model collapsing onto a handful of predicted classes
  (mode collapse from an undertrained, freshly-initialized head).
- **Takeaway**: naively unfreezing the whole backbone does **not** improve on the frozen
  baseline given ~1000 training clips for a 7M-param ResNet pretrained on a different
  (Carnatic) repertoire — it mostly adds training instability within a tractable epoch
  budget. A fairer comparison would likely need: warm-starting the head from Stage 3's
  trained head (`outputs/full50/head_checkpoint.pt`) instead of random init, a higher
  backbone LR with LR warmup, and/or many more epochs (each epoch here takes ~60s, so a much
  longer run is feasible if wanted) plus a less aggressive (or loss-smoothed) early-stopping
  criterion.

**Saved artifacts** (local only, nothing pushed):
- `outputs/full50_finetuned/train_waveforms_8k.pkl` — cached 8kHz-resampled mono waveforms
  for all 1161 train clips (speeds up any rerun).
- `outputs/full50_finetuned/training_curves.png`, `outputs/full50_finetuned/test_confusion_matrix.png`.
- `outputs/full50_finetuned/finetuned_checkpoint.pt` — full model state_dict + config + best
  epoch/val metrics.
- `outputs/full50_finetuned/model/` — full assembled `RaagResNetForAudioClassification`
  (`config.json`, `model.safetensors`, `preprocessor_config.json`, `freeze_backbone=False`).

### Stage 5 — warm-started fine-tunes, all 50 classes (`07_finetune_warmstart_full.py`,
### `08_finetune_warmstart_lastlayer.py`, shared helpers in `finetune_common.py`)

Stage 4's takeaway was that the random head init was the main source of instability. Both
Stage 5 variants instead warm-start the head **and** the `feat_mean`/`feat_scale` scaler from
Stage 3's trained head (`outputs/full50/head_checkpoint.pt`, loaded via the existing
`load_head_weights`) before unfreezing backbone layers. Same train_inner(986)/val(175) split
as Stage 3/4 (same seed=0 `StratifiedShuffleSplit`), same 5s-random-crop / center-crop-val /
full-clip-test methodology as Stage 4. New shared code factored into `finetune_common.py`
(waveform caching now shared at `outputs/_cache/train_waveforms_8k.pkl`, dataset, head
builder, train/eval loop, plotting, confusion matrix); `06_finetune_full_model.py` and its
`outputs/full50_finetuned/` are untouched.

**Stage 5a — warm-started full unfreeze (`outputs/full50_warm_finetuned/`)**: same as Stage 4
(entire backbone + head trainable) but head/scaler warm-started from Stage 3, with gentler
LRs (backbone 3e-5, head 1e-4, vs Stage 4's 1e-5/1e-3 from-scratch). Much more stable than
Stage 4 — train loss/acc improve smoothly (loss 3.50→3.02, acc 0.12→0.24 over 13 epochs) — but
val is still noisy and early-stops quickly: best epoch=2 (val_loss=3.709, val_acc=0.114,
val_f1=0.083), stopped at epoch 12. **Test: accuracy 0.098 (9/92), macro-F1 0.086** —
essentially on par with Stage 3 (0.109/0.102), slightly lower. Warm-starting fixed the
instability/collapse from Stage 4 but full-backbone fine-tuning still doesn't beat the frozen
baseline within a tractable budget — val starts degrading almost immediately (best epoch=2),
suggesting the full 7M-param backbone starts overfitting/drifting from the pretrained Carnatic
features very quickly even at low LR.

**Stage 5b — warm-started, only the last residual block unfrozen
(`outputs/full50_warm_lastlayer_finetuned/`)**: in jeevster's original architecture, the
backbone's last residual block (`res_blocks[9]`) feeds directly into `fc1: Linear(300→150)`
(dropped in our port). "The last layer" = `res_blocks[9]`, NOT what Stage 3 trained — Stage 3
trained an entirely new head on top of a fully-frozen backbone, whereas this stage fine-tunes
the backbone's own last residual block (kept frozen, in eval mode, in Stages 0-3) together
with the head. `conv_first` + `res_blocks[0..8]` stay frozen (`requires_grad=False`, eval
mode, BN stats untouched); only `res_blocks[9]` + head are trainable (564K / 5.49M params,
~10%), both at lr=1e-4 (last block wd=1e-4, head wd=0). Each epoch ~24-29s (much cheaper than
Stage 4/5a since gradients don't need to propagate past `res_blocks[9]`).
- **This is the best result so far.** Train loss decreased steadily and smoothly (3.11→2.49
  over 39 epochs, train acc 0.21→0.40, train macro-F1 0.20→0.40) without the wild train-side
  jumps of 5a. Val loss/acc oscillate epoch-to-epoch but trend down/up overall; best
  epoch=28 (val_loss=3.363, val_acc=0.206, val_f1=0.162), early-stopped at epoch 38.
- **Test: accuracy 0.163 (15/92), macro-F1 0.132** — better than Stage 3 (0.109/0.102), Stage
  4 (0.054/0.019), and Stage 5a (0.098/0.086). `outputs/full50_warm_lastlayer_finetuned/test_confusion_matrix.png`
  shows correct predictions spread across many classes (vs Stage 4's mode-collapse onto a
  handful of columns).
- **Takeaway**: a small, warm-started fine-tuning surface (last residual block + head) beats
  both "frozen backbone, new head" (Stage 3) and "everything unfrozen" (Stage 4/5a) — enough
  capacity to adapt the most task-specific backbone features to Hindustani audio, without
  enough capacity/LR exposure to destabilize or overfit as fast. Best 50-class result across
  all stages so far.

**Saved artifacts** (local only, nothing pushed), per variant:
- `outputs/full50_warm_finetuned/{training_curves.png,test_confusion_matrix.png,
  finetuned_checkpoint.pt,model/}`
- `outputs/full50_warm_lastlayer_finetuned/{training_curves.png,test_confusion_matrix.png,
  finetuned_checkpoint.pt,model/}`
- `outputs/_cache/train_waveforms_8k.pkl` — shared 8kHz waveform cache (used by 07 and 08).

**Follow-ups (not done):** given Stage 5b's clear win, candidates worth trying next: unfreeze
the last *two* residual blocks (`res_blocks[8:]`) + head; longer patience / more epochs for
5b (still improving at epoch 38 when stopped); a small LR schedule (decay) to tame the val
oscillation seen in 5b's loss curve.

### Stage 6 — Stage 5b follow-ups, all 50 classes (`10_finetune_warmstart_last2blocks.py`,
### `11_finetune_warmstart_lastlayer_longer.py`, `12_finetune_warmstart_lastlayer_lrdecay.py`)

Three follow-ups to Stage 5b's three candidate ideas, run as background jobs. All three reuse
Stage 5b's train_inner(986)/val(175) split, 5s-random-crop / center-crop-val / full-clip-test
methodology, warm-started head + feature scaler (`outputs/full50/head_checkpoint.pt`), and
`finetune_common.py` helpers. None touch `outputs/full50_warm_lastlayer_finetuned` or any
earlier-stage artifacts.

**Stage 6a — unfreeze the last TWO residual blocks (`res_blocks[8:]`) + head
(`outputs/full50_warm_last2blocks_finetuned/`)**: doubles the trainable surface to
1,106,242 / 5,489,542 params (~20%), both groups at lr=1e-4 (last-blocks wd=1e-4, head wd=0).
Converges much faster than 5b — best epoch=9 (val_loss=3.377, val_acc=0.200, val_f1=0.163),
early-stopped at epoch 19. **Test: accuracy 0.174 (16/92), macro-F1 0.146 — the best result
across all stages so far**, beating Stage 5b (0.163/0.132). The extra unfrozen block gives
more capacity without needing as many epochs to find it, though val performance also degrades
faster after its peak than 5b's did (best at epoch 9 vs 5b's epoch 28).

**Stage 6b — Stage 5b with a longer budget (EPOCHS 40→80, PATIENCE 10→20)
(`outputs/full50_warm_lastlayer_longer_finetuned/`)**: ran the full 80 epochs without
early-stopping. Train metrics kept climbing the whole time (train acc 0.21→0.48, train
macro-F1 0.20→0.48 by epoch 79), but val_loss only marginally improved on its best epoch:
best epoch=60 (val_loss=3.3205, val_acc=0.189, val_f1=0.156) vs 5b's best epoch=28
(val_loss=3.363, val_acc=0.206, val_f1=0.162) — nearly the same val_loss, slightly worse
val_acc/F1. **Test: accuracy 0.141 (13/92), macro-F1 0.110** — worse than 5b despite the
marginally-better val_loss, and worse than 6a. **Takeaway**: more epochs don't help here —
val_loss plateaus by ~epoch 28-60 while train metrics keep climbing (overfitting on the
training_inner split), and the 175-example val set is too small/noisy to reliably pick a
better checkpoint from the extra epochs. The extra training time bought nothing.

**Stage 6c — Stage 5b with cosine LR decay (each param group's lr decays from 1e-4 to 1e-5
over 40 epochs) (`outputs/full50_warm_lastlayer_lrdecay_finetuned/`)**: early-stopped at
epoch 24, best epoch=14 (val_loss=3.381, val_acc=0.183, val_f1=0.154). **Test: accuracy 0.152
(14/92), macro-F1 0.096** — worse than 5b on both metrics, and the worst macro-F1 of the
Stage 5/6 family despite a mid-pack accuracy (suggesting predictions concentrated on fewer
classes). The decay didn't tame the val oscillation (val_loss/acc still swing epoch-to-epoch
in the log) and converged to a worse checkpoint earlier than 5b.

**Updated leaderboard (50-class test set, by macro-F1)**:

| Stage | accuracy | macro-F1 |
|---|---|---|
| 6a — last 2 blocks unfrozen | 0.174 | **0.146** |
| 5b — last block unfrozen | 0.163 | 0.132 |
| 6b — last block, longer budget | 0.141 | 0.110 |
| 3 — frozen backbone, new head | 0.109 | 0.102 |
| 6c — last block, cosine LR decay | 0.152 | 0.096 |
| 5a — full unfreeze, warm-started | 0.098 | 0.086 |
| 4 — full unfreeze, random head | 0.054 | 0.019 |

**Saved artifacts** (local only, nothing pushed), per variant:
- `outputs/full50_warm_last2blocks_finetuned/{training_curves.png,test_confusion_matrix.png,
  finetuned_checkpoint.pt,model/}`
- `outputs/full50_warm_lastlayer_longer_finetuned/{training_curves.png,test_confusion_matrix.png,
  finetuned_checkpoint.pt,model/}`
- `outputs/full50_warm_lastlayer_lrdecay_finetuned/{training_curves.png,test_confusion_matrix.png,
  finetuned_checkpoint.pt,model/}`

**Follow-ups (not done):** Stage 6a is the new best result. Candidates worth trying next if
pursued further: unfreeze the last *three* residual blocks (`res_blocks[7:]`) + head, to see
whether the "wider unfrozen surface converges faster" trend from 6a continues or reverses;
combine 6a's wider unfreeze with a lower patience (it peaked at epoch 9 vs 5b's epoch 28, so
patience=10 may already be near-optimal for that configuration but could be tightened
further).
