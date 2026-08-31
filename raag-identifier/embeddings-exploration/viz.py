# Visualization for embedding exploration.
#
# Three levels of analysis (in order of insight for melody):
#
#   1. Clip-level scatter  — each point = one clip (pooled vector), colored by label.
#                            Quick sanity check: do same-raag clips cluster?
#
#   2. Chunk trajectory    — each point = one chunk, lines connect chunks from the
#                            same clip in time order. Reveals temporal drift and
#                            within-clip structure without destroying sequence info.
#
#   3. Self-similarity     — pairwise cosine similarity between all chunks of a single
#                            clip. Repeated melodic sections appear as off-diagonal
#                            blocks. More informative than global scatter for melody.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config

# Tagged with the chunk size so this run's plots don't overwrite Attempt 1's
# 10s plots (plots/<model_name>/...).
PLOTS_DIR = Path(__file__).parent / "plots" / "2s"


def _finish(fig, save: bool, filename: str):
    """Either save the figure to disk or display it interactively."""
    if save:
        out = PLOTS_DIR / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  saved → {out}")
    else:
        plt.show()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _iter_npz(model_name: str, split: str):
    """Yield (dataset_idx, npz_data) sorted numerically by index."""
    out_dir = config.OUTPUT_DIR / model_name
    paths = sorted(out_dir.glob(f"{split}_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    for path in paths:
        idx = int(path.stem.split("_")[1])
        yield idx, np.load(path)


def _balanced_paths(model_name: str, split: str, all_labels: np.ndarray, max_clips: int):
    """
    Return up to max_clips paths sampled in round-robin order across label classes.
    Fixes the lexicographic-sort bug where first-N files fall on only 1–2 labels.
    """
    from collections import defaultdict
    by_label = defaultdict(list)
    for path in sorted(
        (config.OUTPUT_DIR / model_name).glob(f"{split}_*.npz"),
        key=lambda p: int(p.stem.split("_")[1])
    ):
        idx = int(path.stem.split("_")[1])
        by_label[all_labels[idx]].append(path)

    selected, iters = [], {lbl: iter(paths) for lbl, paths in by_label.items()}
    while len(selected) < max_clips:
        added = False
        for lbl in sorted(iters):
            if len(selected) >= max_clips:
                break
            try:
                selected.append(next(iters[lbl]))
                added = True
            except StopIteration:
                pass
        if not added:
            break
    return selected


def _longest_clip_idx(model_name: str, split: str) -> int:
    """Return the dataset index of the clip with the most saved chunks."""
    best_idx, best_n = 0, 0
    for path in (config.OUTPUT_DIR / model_name).glob(f"{split}_*.npz"):
        n = np.load(path)["chunks"].shape[0]
        if n > best_n:
            best_n = n
            best_idx = int(path.stem.split("_")[1])
    return best_idx


def load_clip_vectors(model_name: str, split: str, kind: str = "clip_mean"):
    """Return (matrix [n_clips, d], label_indices [n_clips]) for a given pooling kind."""
    embs, idxs = [], []
    for idx, data in _iter_npz(model_name, split):
        embs.append(data[kind])
        idxs.append(idx)
    return np.array(embs), np.array(idxs)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def _umap_2d(embeddings: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(random_state=42).fit_transform(embeddings)


def _tsne_2d(embeddings: np.ndarray, pca_dim: int = 50) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if embeddings.shape[1] > pca_dim:
        embeddings = PCA(n_components=pca_dim).fit_transform(embeddings)
    return TSNE(n_components=2, random_state=42).fit_transform(embeddings)


# ── Level 1: clip-level scatter ───────────────────────────────────────────────

def plot_clip_scatter(model_name: str, split: str, all_labels: np.ndarray,
                      method: str = "umap", kind: str = "clip_mean", save: bool = False):
    """2D scatter of pooled clip embeddings, colored by raag label."""
    embs, idxs = load_clip_vectors(model_name, split, kind)
    labels = all_labels[idxs]

    coords = _umap_2d(embs) if method == "umap" else _tsne_2d(embs)

    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=50)
    plt.title(f"{model_name} | {split} | {method} ({kind})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    _finish(fig, save, f"{model_name}/{split}_clip_scatter_{method}_{kind}.png")


# ── Level 2: chunk trajectory map ────────────────────────────────────────────

def plot_chunk_trajectories(model_name: str, split: str, all_labels: np.ndarray,
                             max_clips: int = 30, save: bool = False):
    """
    All chunks from up to max_clips clips projected to 2D.
    Chunks from the same clip are connected by a line in time order.
    Clips are sampled balanced across label classes to avoid lexicographic bias.
    """
    all_embs, clip_sizes, clip_labels = [], [], []

    paths = _balanced_paths(model_name, split, all_labels, max_clips)
    for path in paths:
        idx    = int(path.stem.split("_")[1])
        chunks = np.load(path)["chunks"]   # (n_chunks, d)
        all_embs.append(chunks)
        clip_sizes.append(len(chunks))
        clip_labels.append(all_labels[idx])

    coords   = _umap_2d(np.concatenate(all_embs, axis=0))
    palette  = sns.color_palette("tab10", n_colors=len(set(clip_labels)))
    color_of = {lbl: palette[i] for i, lbl in enumerate(sorted(set(clip_labels)))}

    fig, ax = plt.subplots(figsize=(12, 9))
    pos = 0
    for n, lbl in zip(clip_sizes, clip_labels):
        c = coords[pos : pos + n]
        ax.plot(c[:, 0], c[:, 1], color=color_of[lbl], alpha=0.4, linewidth=0.8)
        ax.scatter(c[:, 0], c[:, 1], color=color_of[lbl], s=18)
        pos += n

    ax.set_title(f"{model_name} | {split} | chunk trajectories (n={len(paths)} clips)")
    plt.tight_layout()
    _finish(fig, save, f"{model_name}/{split}_chunk_trajectories.png")


# ── Level 3: self-similarity matrix ──────────────────────────────────────────

def plot_self_similarity(model_name: str, split: str, clip_idx: int | None = None,
                          save: bool = False):
    """
    Cosine similarity between every pair of chunks within a single clip.
    Repeated melodic phrases show up as off-diagonal bright blocks.
    If clip_idx is None, auto-selects the clip with the most chunks.
    """
    if clip_idx is None:
        clip_idx = _longest_clip_idx(model_name, split)
        print(f"  [{model_name}] selfsim: using longest clip train_{clip_idx}")
    path   = config.OUTPUT_DIR / model_name / f"{split}_{clip_idx}.npz"
    chunks = np.load(path)["chunks"]   # (n_chunks, d)

    norms  = np.linalg.norm(chunks, axis=1, keepdims=True)
    normed = chunks / np.clip(norms, 1e-8, None)
    sim    = normed @ normed.T           # (n_chunks, n_chunks)

    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(sim, vmin=0, vmax=1, cmap="magma", square=True)
    plt.title(f"{model_name} | {split}[{clip_idx}] | chunk self-similarity")
    plt.xlabel("chunk index")
    plt.ylabel("chunk index")
    plt.tight_layout()
    _finish(fig, save, f"{model_name}/{split}_{clip_idx}_selfsim.png")
