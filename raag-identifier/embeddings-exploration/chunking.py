# All chunking and pooling strategies.
# embed.py calls chunk_audio to split a clip, then one of the pool_* functions
# per saved representation. Add new pooling strategies here as needed.

import numpy as np


def chunk_audio(array: np.ndarray, sr: int, chunk_size_s: float, overlap: float) -> list[np.ndarray]:
    """Split a 1-D audio array into overlapping fixed-length chunks.

    If the clip is shorter than one chunk, it is returned as-is (single chunk).
    """
    chunk_len = int(chunk_size_s * sr)
    hop_len = int(chunk_len * (1 - overlap))
    starts = list(range(0, len(array) - chunk_len + 1, hop_len))
    if not starts:
        return [array]
    return [array[s : s + chunk_len] for s in starts]


# ── Pooling (chunk_embeddings: np.ndarray of shape [n_chunks, d]) ─────────────

def pool_mean(chunk_embeddings: np.ndarray) -> np.ndarray:
    """Mean across chunks. Good for retrieval; loses temporal order."""
    return np.mean(chunk_embeddings, axis=0)


def pool_rich(chunk_embeddings: np.ndarray) -> np.ndarray:
    """Concat(mean, std, max) — richer summary, 3× the dimension."""
    return np.concatenate([
        np.mean(chunk_embeddings, axis=0),
        np.std(chunk_embeddings, axis=0),
        np.max(chunk_embeddings, axis=0),
    ])
