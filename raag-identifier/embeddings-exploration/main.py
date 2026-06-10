"""
Embeddings Exploration — entry point
=====================================

SETUP
-----
Activate the poetry env:
    source $(poetry env info --path)/bin/activate

FULL EXPERIMENT CHECKLIST
--------------------------
1. Edit config.py
       MODELS_TO_RUN   — which models to run
       LABEL_INDICES   — which raag indices to process (start with range(0, 5))
       CHUNK_SIZE_S    — seconds per chunk (10 s is a good default)
       CHUNK_OVERLAP   — fraction overlap (0.5 = 50%)

2. Compute embeddings  (skips clips that are already saved)
       python main.py embed

3. Visualize
       python main.py viz clip     # clip-level UMAP (level 1)
       python main.py viz traj     # chunk trajectory map (level 2)
       python main.py viz selfsim  # self-similarity matrix for first clip (level 3)
       Add --save to write PNGs to plots/<model_name>/ instead of showing interactively.

4. Iterate
       - Add more models to MODELS_TO_RUN and re-run embed (existing files are skipped)
       - Expand LABEL_INDICES to cover more raags
       - Compare plots across models side-by-side

NOTES
-----
- All outputs land in embeddings-exploration/outputs/<model_name>/
- Each .npz stores: chunks [n_chunks, d], clip_mean [d], clip_rich [3d]
- MAEST / MuQ / MusicFM are stubbed — implement their model files before adding to run
"""

import sys
import numpy as np
from datasets import load_dataset

import config
from embed import embed_dataset
from viz import plot_clip_scatter, plot_chunk_trajectories, plot_self_similarity


def run_embed():
    for model_name in config.MODELS_TO_RUN:
        print(f"\n=== {model_name} ===")
        embed_dataset(model_name)


def run_viz(subcommand: str, save: bool = False):
    ds     = load_dataset(config.DATASET_ID, revision=config.DATASET_REVISION)
    labels = np.array(ds["train"]["label"])

    for model_name in config.MODELS_TO_RUN:
        if subcommand == "clip":
            plot_clip_scatter(model_name, "train", labels, save=save)
        elif subcommand == "traj":
            plot_chunk_trajectories(model_name, "train", labels, save=save)
        elif subcommand == "selfsim":
            plot_self_similarity(model_name, "train", clip_idx=None, save=save)
        else:
            print(f"Unknown viz subcommand '{subcommand}'. Options: clip | traj | selfsim")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    args = sys.argv[1:]
    save = "--save" in args
    args = [a for a in args if a != "--save"]

    cmd = args[0]
    if cmd == "embed":
        run_embed()
    elif cmd == "viz" and len(args) >= 2:
        run_viz(args[1], save=save)
    else:
        print(__doc__)
        sys.exit(1)
