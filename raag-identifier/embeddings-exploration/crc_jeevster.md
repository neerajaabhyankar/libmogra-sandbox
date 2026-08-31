# `carnatic-raga-classifier-jeevster` — what it is and how to get embeddings out of it

Source: mirrored from https://huggingface.co/spaces/jeevster/carnatic-raga-classifier (someone else's
half-finished HF Space). Symlinked into this repo as `raag-identifier/carnatic-raga-classifier-jeevster`
— **read-only, do not edit**.

## What's in the folder

| File | Purpose |
|---|---|
| `models.py` | Three candidate architectures: `BaseRagaClassifier` (plain CNN), `ResNetRagaClassifier` (1D-ResNet CNN), `Wav2VecTransformer` (wav2vec2 encoder + linear head). Only **ResNet** has a checkpoint. |
| `config0.yaml` | Hyperparams for the saved checkpoint: `model: resnet`, `clip_length: 30` (seconds), `sample_rate: 8000`, `n_blocks: 10`, `n_channel: 300`, `stride: 16`, `max_pool_every: 1`, `num_classes: 150`. |
| `ckpts/best_ckpt.tar` | Trained weights for `ResNetRagaClassifier` (epoch 53, ~66 MB). `state_dict` keys match `models.py` exactly — loads cleanly with `params.input_channels=2`. |
| `raga2label0.json` | A raga-name → int label dict, **528 entries** (0–527). This is the bug the user flagged: `config0.yaml` says `num_classes: 150`, but the saved label map has 528 entries and was never truncated/regenerated to match the 150-class checkpoint. The `fc1` layer is genuinely `Linear(300, 150)`, so labels ≥150 in this JSON cannot correspond to real output classes — the mapping is unusable for turning logits back into raga names without redoing `dataloader.get_raga2label()` against whatever `num_files_per_raga.json` produced the original 150-class ordering (that file isn't present). |
| `dataloader.py` | Training-time dataset: loads 30 s stereo clips at 8 kHz, per-channel-normalizes (`(x - mean) / (std + 1e-5)` over the time axis), pads short clips with zeros. |
| `inference.py` | `Evaluator` class: loads the model + `raga2label0.json`, runs sliding 30 s windows over a clip, averages softmax probabilities, returns top-k raga names. Has a typo bug (`params.clip_lenght`) in the zero-pad path that would crash on any clip shorter than 30 s. |
| `main.py` | Tiny smoke-test script that instantiates `Evaluator` and runs `inference()` on one local file. |

**Bottom line on "no label mappings"**: the checkpoint is a perfectly good 150-way classifier, but the
raga-name ↔ label-index mapping needed to interpret its 150 output logits doesn't exist in this folder.
Since we only want **embeddings**, not predictions, this doesn't block us — we never touch `fc1`'s output
or `raga2label0.json`.

## Architecture (`ResNetRagaClassifier`, the only one with weights)

```
input: (batch, 2, T)                    # stereo, 8 kHz, per-channel normalized

conv_first: Conv1d(2 → 300, kernel=80, stride=16) + BatchNorm1d + ReLU
                                          # T -> floor((T-80)/16)+1

10 × ResidualBlock(300 channels, kernel=3):
    conv_block1: Conv1d(300→300, k=3, "same") + BN + ReLU
    conv_block2: Conv1d(300→300, k=3, "same") + BN + ReLU
    out = conv_block2(conv_block1(x)) + x        # residual add
  -- max_pool_every=1, so EVERY block is followed by MaxPool1d(2) --
                                          # length halved 10 times (÷1024 total)

avg_pool1d(x, x.shape[-1])              # global average pool over whatever
                                          # time dimension remains -> (batch, 300, 1)
permute -> (batch, 1, 300)

fc1: Linear(300 -> 150)
log_softmax(dim=-1)                     # (batch, 1, 150)
```

~25 conv layers total, 300 channels throughout, ~7M params. The conv stem (k=80, stride=16) plus the
10 halving max-pools give an effective ÷1024 temporal downsampling before the global average pool — this
is what makes the network output a *fixed-size* (300,) vector regardless of input length (the avg-pool
kernel size is `x.shape[-1]`, computed dynamically).

## Best layer for embeddings: the `fc1` input (300-dim, global-avg-pooled)

This is the standard "penultimate layer" embedding for a classifier: everything the network learned in
order to discriminate between 150 ragas, collapsed to one 300-dim vector per clip, *before* the final
linear projection throws away dimensions to get down to 150 logits. It's the highest-level
melody/timbre-relevant representation the model has — the conv blocks before it are progressively
larger-receptive-field feature extractors, and `fc1`'s output is just a re-projection of this vector into
class-score space (which we don't want).

Implementation: register a forward hook on `model.fc1` and capture its input (shape `(batch, 1, 300)` →
squeeze to `(300,)`).

## Input-length constraints (measured empirically)

The model was trained on 30 s @ 8 kHz clips (240,000 samples → 14,996 samples after the conv stem → 14
after 10 halvings → avg-pooled to 1). Because the avg-pool is dynamic, **shorter inputs also produce a
valid (300,) output — down to a point**:

| Input duration (8 kHz) | Samples | Result |
|---|---|---|
| 2 s | 16,000 | **crashes** — `max_pool1d() Invalid computed output size: 0` (the 10th halving hits a length-1 tensor) |
| 5 s | 40,000 | works — feature length collapses to 2 before the final avg-pool |
| 10 s / 15 s / 30 s / 45 s | 80,000+ | all work, output is always (300,) |

So there's a hard floor somewhere between 2 s and 5 s @ 8 kHz. Additionally, **all the BatchNorm running
stats were calibrated on 30 s clips** (where the avg-pool sees ~14 positions); feeding much shorter clips
means the avg-pool sees only 1–2 positions, which is a mild train/test mismatch but not catastrophic
(unlike CLAP's 10 s zero-padding problem, which is ~80% padding for 2 s chunks).

## Practical implication for this dataset

Our Hindustani clips are mostly **5–10 s** (see Attempt 1 findings in `plan.md`). Given:
- 2 s chunks (our current `CHUNK_SIZE_S`) would crash this model,
- the model has no meaningful "trajectory" structure — it was designed to consume an entire clip and
  output one vector,

the right unit of analysis here is **one embedding per whole clip** (resample to 8 kHz, mono→stereo,
per-channel normalize, zero-pad up to a 5 s floor if needed, no chunking). That gives us **points in
embedding space, one per clip** — a Level-1 clip scatter only. Chunk trajectories (Level 2) and
self-similarity (Level 3) don't apply: there's exactly one "chunk" per clip by construction.
