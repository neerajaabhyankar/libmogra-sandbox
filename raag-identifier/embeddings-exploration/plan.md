# Embeddings Exploration — Plan

Goal: run several SOTA open-source audio/music embedding models on every clip in the dataset, save the outputs, then look for melody-correlated structure in the embedding space.

---

## Models

Priority order for a MacBook 16 GB:

| Model | Size | Mac 16 GB | HuggingFace / repo |
|---|---|---|---|
| MERT-v1-95M | 95 M | Yes, comfortable | `m-a-p/MERT-v1-95M` |
| MERT-v1-330M | 330 M | Likely fine with chunking | `m-a-p/MERT-v1-330M` |
| MAEST | ~86 M | Yes, friendly | `MTG/discogs-maest-*` |
| LAION-CLAP (music) | ~150 M | Yes | `laion/larger_clap_music` |
| MuQ | ~300 M | Marginal — CPU/MPS with care | `mulab-ai/MuQ` |
| MusicFM | ~300 M | Probably fine | `minzwon/musicfm` (0.3B ckpt) |
| MelodySim | MERT-based | Same as MERT backend | find checkpoint via paper repo |

Skip for now: **MuQ-MuLan** (700 M) and **Jukebox** (multi-GB, impractical on 16 GB).

---

## Chunk size: thinking about melody

Notes in Hindustani music change on the order of 0.2–2 s depending on the passage (fast tans vs. slow alaap). A melodic phrase (pakad, characteristic motif) typically spans 1–5 s. This has direct implications for chunk size:

| Chunk size | What it captures | Problem |
|---|---|---|
| 10 s | Multiple full phrases | Mean-pooling over 10 s averages together several melodic ideas; self-similarity matrices degenerate on short clips |
| 2–3 s | One melodic phrase or motif | Matches melody timescale; gives far more chunks per clip |
| < 1 s | Individual notes | Too noisy; may not give models enough context |

**Recommended for Attempt 2: 2 s chunks, 50% overlap (1 s hop).**

Important caveat by model:
- **MERT** processes at 24 kHz and handles variable-length inputs naturally. 2 s = 48 000 samples — works fine.
- **CLAP** has a fixed ~10 s input window; inputs shorter than that are zero-padded. Passing 2 s chunks to CLAP means 80% of the input is silence, which will badly distort the embedding. For CLAP, keep 5–10 s chunks or accept that it is not suited for fine-grained melody analysis.
- **MelodySim** (MERT-based): same as MERT, 2 s chunks fine.

---

## Output format

Precompute once, save to disk. Layout:

```
embeddings-exploration/
  outputs/
    <model_name>/
      <split>_<idx>.npz      # keys: chunks [n_chunks,d], clip_mean [d], clip_rich [3d]
  plots/
    <model_name>/
      *.png
```

---

## Visualization plan

Three levels, in order of insight for melody:

1. **Clip-level UMAP** — each point = one clip, embedding = `clip_mean`. Color by raag label. Quick sanity check; expected to show weak separation since most models don't optimise for melody.

2. **Chunk trajectory map** — each point = one chunk, lines connect chunks from the same clip in time order. Reveals temporal drift and within-clip structure. Must sample **balanced across label classes** (not first-N files, which has a lexicographic sort bug that concentrates on a subset of labels).

3. **Self-similarity matrices** — pairwise cosine similarity between all chunks of one clip. Repeated melodic sections appear as off-diagonal blocks. Only meaningful for clips with ≥ 6–8 chunks, i.e., clips ≥ ~9 s at 2 s / 50% overlap. Auto-select the longest clip.

---

## Key caveats

- Most models learn timbre, genre, production, or loudness before melody. MelodySim is the only one explicitly trained for melodic similarity; treat others as baselines.
- Embeddings are generally **not transposition-invariant**.
- Mean-pooling destroys temporal order — prefer chunk trajectories and self-similarity for melody questions.
- Validate UMAP clusters by listening to nearest neighbors, not just by visual inspection.
- MuQ weights are CC-BY-NC 4.0 — non-commercial only.

---

## Compute notes

- batch size 1–4, float16 where supported, CPU fallback acceptable (slow)
- Run on MPS (Apple Silicon) if the model supports it
- Precompute all embeddings once; never rerun a model just to visualize

---

---

## What We Tried

### Attempt 1 — CLAP + MERT-95M, 10 s chunks

**Config:** `CHUNK_SIZE_S=10`, `CHUNK_OVERLAP=0.5`, labels 0–4 (AheerBhairav, AlhaiyaBilawal, Bageshree, Bahar, Bairagi), train split.

**Findings:**

| Issue | Detail |
|---|---|
| Chunks too sparse | 71% of clips produced only 1 chunk (most clips < 15 s). Self-similarity matrices were 1×1 or 2×2 — meaningless. |
| Self-similarity was degenerate | Even the longest clips had only 5 chunks, so no repeating structure could be seen. |
| Clip scatter: no separation | Both CLAP and MERT UMAP showed all 5 raag classes intermixed. Expected — these models don't optimise for melody. |
| Trajectory plot: only 2 label classes | Bug: `glob` returns files in lexicographic order (`train_0, train_1, train_10, train_100...`), so the first N files land on labels 0 and 4 only, skipping 1, 2, 3. Fixed in code: now uses balanced per-label sampling. |
| CLAP unsuitable for short chunks | CLAP has a ~10 s fixed input window; chunks shorter than that get zero-padded, distorting the embedding. |

**Conclusion:** 10 s chunks are too coarse for melody. The dataset clips are shorter than expected — the median is closer to 5–10 s, not 25 s.

---

### Attempt 2 — MERT-95M + MelodySim, 2 s chunks

**Changes from Attempt 1:**
- `CHUNK_SIZE_S = 2.0` (down from 10) — matches melody timescale
- Add MelodySim (melody-specific model, directly relevant to the goal)
- Drop CLAP from fine-grained chunk analysis (keep for clip-level UMAP only as semantic baseline)
- Trajectory and selfsim plots now use balanced label sampling and auto-select the longest clip

**Run config:** `CHUNK_SIZE_S=2.0`, `CHUNK_OVERLAP=0.5`, `MODELS_TO_RUN=["mert-95m","melodysim"]` (CLAP excluded), labels 0–4, train split. Outputs in `outputs/2s/`, plots in `plots/2s/` (kept separate from Attempt 1's `outputs/`/`plots/` so neither overwrites the other).

**Findings:**

| Aspect | MERT-95M | MelodySim |
|---|---|---|
| Chunks per clip | train_53 (longest): 31 chunks (vs. 5 at 10 s) | same — 31 chunks |
| Clip-level UMAP | Still no visible class separation; one big diffuse blob | Also no clean class separation, but clips spread into a looser, more elongated manifold with a couple of small isolated sub-clusters |
| Chunk trajectories | Each clip forms a **tight, well-separated cluster** with visible internal trajectory wiggle — chunks from the same clip stay close in embedding space (timbre/recording-condition dominated) | Chunks from different clips are **heavily intermixed** across the whole plot — much higher chunk-to-chunk variance, less dominated by clip identity |
| Self-similarity (train_53, 31×31) | Mostly **flat and high** (~0.8–0.95 everywhere); only mild off-diagonal dips (e.g. chunk 11, chunk 26) — embeddings barely change over time | **Much higher contrast** (full 0–1 range), with clear block structure: chunks 0–10 form one loose block, chunks 22–30 another, and chunk 12 is a near-zero-similarity outlier to everything. Looks like it's distinguishing different sections of the performance |

**Conclusion:** 2 s chunks fixed the sparsity problem (31 chunks vs. 5). MelodySim's self-similarity matrix now shows real block structure plausibly corresponding to distinct melodic sections — much more promising for melody analysis than MERT, whose chunk embeddings stay nearly constant across a clip (likely capturing timbre/recording identity rather than melodic content). Possible next steps: listen to train_53 against the MelodySim block boundaries (~chunk 11–12 and ~chunk 22) to check whether they correspond to actual melodic transitions; try chunk-level UMAP (not just clip-level) to see if MelodySim chunks cluster by melodic phrase across clips.

---

### Attempt 3 — CRC-Jeevster (Carnatic raga classifier), clip-level embeddings only

**Goal:** add a third embedding model — a pretrained 1D-ResNet raga classifier
(`carnatic-raga-classifier-jeevster`, symlinked into this repo) — and extract its
penultimate-layer (300-dim) embedding per clip. Background and architecture details
written up in `crc_jeevster.md`.

**Why it's different from MERT/MelodySim:**
- Trained on **30 s @ 8 kHz stereo** clips, per-channel normalized; has a hard
  minimum input length around 2–5 s @ 8 kHz (shorter inputs crash in `max_pool1d`).
- Its global-avg-pool collapses time to a single vector regardless of input length —
  so it naturally produces **one embedding per clip**, not a temporal sequence.
- Our dataset clips are mostly 5–10 s, well under the 30 s training length, so we
  treat each clip as a single unit rather than trying to chunk it like MERT/MelodySim.

**Plan:**
1. New file `models/crc_jeevster.py`:
   - `load()`: import `ResNetRagaClassifier` from the symlinked jeevster folder,
     build it with the params from `config0.yaml` (`input_channels=2, n_channel=300,
     stride=16, n_blocks=10, max_pool_every=1, num_classes=150`), load
     `ckpts/best_ckpt.tar`, register a forward hook on `fc1` to capture its input.
   - `embed(audio_array, sr)`: resample to 8 kHz, mono→stereo, per-channel normalize
     (matching `dataloader.py`), zero-pad up to a 5 s floor (40,000 samples) if
     shorter, forward pass, return the captured (300,) `fc1`-input vector.
   - Mark this embedder as **whole-clip** (no chunking) — add a `whole_clip = True`
     class attribute on `BaseEmbedder` (default `False`) that `embed.py` checks to
     decide whether to call `chunk_audio` or just pass `[array]`.
2. Register `"crc-jeevster"` in `models/__init__.py` REGISTRY.
3. Add `"crc-jeevster"` to `config.MODELS_TO_RUN` (alongside mert-95m, melodysim),
   same `LABEL_INDICES = range(0, 5)`, train split. Outputs land in
   `outputs/2s/crc-jeevster/` (dir name kept consistent with this attempt's run,
   even though this model isn't actually chunked at 2 s).
4. Run `python main.py embed`, then `python main.py viz clip --save` for
   `crc-jeevster` only — **clip-level UMAP scatter (Level 1) is the only
   meaningful plot here**; trajectory/self-similarity plots are skipped (n_chunks=1
   per clip by construction, so they'd be degenerate).

**Open question to validate after running:** does this raga-classifier's embedding
space show *more* class separation by raag than MERT/MelodySim (it was explicitly
trained to discriminate ragas, albeit Carnatic ones on a different 150-class label
set) — even though it was never trained on Hindustani audio?

**Setup fix:** the symlink at `raag-identifier/carnatic-raga-classifier-jeevster` was
broken (`../../carnatic-raga-classifier-jeevster`, two levels up — pointed outside
the repo entirely). Fixed to `../carnatic-raga-classifier-jeevster` (one level up,
to `icm-shruti-analysis/carnatic-raga-classifier-jeevster`). Also had to load
jeevster's `models.py` via `importlib` under a distinct module name
(`jeevster_models`), since the name `models` collides with this project's own
`models/` package.

**Findings:**

| Aspect | Result |
|---|---|
| Dataset clip durations (label 0–4, train) | min 4 s, max 32 s, median 12 s — only 3/123 clips fall under the model's 5 s zero-pad floor, so padding has minimal overall effect |
| Clip-level UMAP (`clip_mean`, = the raw 300-dim embedding since n_chunks=1) | Same story as MERT/MelodySim: one large diffuse blob, **no visible raag separation** for labels 0–4. A small isolated group of ~9 points splits off (train_27/28, 33/34, 73/74, 111/112/115) — these are pairs/triples of *different* labels with matching durations (11–20 s each), so the isolation looks like a shared-source-recording effect, not a duration artifact. |
| `clip_rich` (mean/std/max) | **Degenerate for this model** — with n_chunks=1, std=0 and max=mean, so `clip_rich` = `[emb, zeros(300), emb]`. Dropped this plot; only `clip_mean` is meaningful here. |
| Chunk trajectories / self-similarity | Not computed — n_chunks=1 per clip by construction (whole_clip=True), so both would be degenerate (single point / 1×1 matrix). |

**Conclusion:** Even a model explicitly trained as a raga classifier (just on a
different repertoire/label set) shows no visible class separation in 2D UMAP for
this 5-raag Hindustani subset — consistent with MERT and MelodySim. This is some
evidence that the *visualization* (clip-level UMAP on 5 classes, ~25 clips/class) may
be the bottleneck rather than any one embedding model: either raag identity isn't
the dominant axis of variation in 300-d/768-d space at this scale, or 2D UMAP from
~25 points/class is too sparse to reveal it. Possible next steps: (1) try the
`clip_rich`-equivalent for crc-jeevster by chunking each clip into multiple
≥5 s windows (e.g. 10 s, 50% overlap) instead of whole-clip, to get a `std`/`max`
signal and more points per clip; (2) increase `LABEL_INDICES` to more raags and/or
more clips per raag to see if separation emerges with more data; (3) try a
supervised metric (e.g. k-NN accuracy on raag label using these embeddings) instead
of relying on visual UMAP separation, which can hide structure that exists in
higher dimensions.
