# Core embedding loop.
#
# For each configured model × split × clip:
#   - Skip if clip label is not in config.LABEL_INDICES
#   - Skip if output .npz already exists (safe to re-run / resume)
#   - Chunk the audio, embed each chunk, pool, save
#
# Each .npz contains three arrays:
#   chunks    : (n_chunks, d)  — full temporal sequence; needed for viz levels 2 & 3
#   clip_mean : (d,)           — mean-pooled; good for retrieval / clip-level UMAP
#   clip_rich : (3d,)          — concat(mean, std, max); richer but larger

import numpy as np
from datasets import load_dataset, Audio as HFAudio

import config
from chunking import chunk_audio, pool_mean, pool_rich
from models import get_embedder


def _out_path(model_name: str, split: str, idx: int):
    return config.OUTPUT_DIR / model_name / f"{split}_{idx}.npz"


def embed_dataset(model_name: str):
    embedder = get_embedder(model_name)
    embedder.load()
    print(f"[{model_name}] model loaded")

    ds = load_dataset(config.DATASET_ID, revision=config.DATASET_REVISION)
    # Preserve native sampling rate; each model resamples internally as needed.
    ds = ds.cast_column("audio", HFAudio(sampling_rate=None))

    for split in ds.keys():
        out_dir = config.OUTPUT_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(ds[split])
        for idx, sample in enumerate(ds[split]):
            if sample["label"] not in config.LABEL_INDICES:
                continue

            out_path = _out_path(model_name, split, idx)
            if out_path.exists():
                continue  # resume-safe: skip already-computed clips

            array = sample["audio"]["array"]
            sr    = sample["audio"]["sampling_rate"]

            if embedder.whole_clip:
                chunks = [array]
            else:
                chunks = chunk_audio(array, sr, config.CHUNK_SIZE_S, config.CHUNK_OVERLAP)
            chunk_embs = np.array([embedder.embed(c, sr) for c in chunks])  # (n_chunks, d)

            np.savez(
                out_path,
                chunks=chunk_embs,
                clip_mean=pool_mean(chunk_embs),
                clip_rich=pool_rich(chunk_embs),
            )

            if idx % 10 == 0:
                print(f"  [{split}] {idx}/{total}")
