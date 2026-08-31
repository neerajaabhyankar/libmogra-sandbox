# All experiment knobs live here. Change these before running main.py.

from pathlib import Path

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_ID = "neerajaabhyankar/hindustani-raag-small"
DATASET_REVISION = "0dfb021e54e0e7489b90a47e23ef15f34fa740ec"

# ── Labels ────────────────────────────────────────────────────────────────────
# Only process clips whose label index falls in this list.
# Full dataset has 50 raags; start small and expand.
LABEL_INDICES = list(range(0, 5))

# ── Models ────────────────────────────────────────────────────────────────────
# Valid names: "clap", "mert-95m", "mert-330m", "maest", "muq", "musicfm", "crc-jeevster"
# See models/ for which are fully implemented vs stubbed.
# CLAP dropped for the 2s run: it has a fixed ~10s input window and 2s chunks
# would be 80% zero-padding (see plan.md "Attempt 2"). Its 10s embeddings/plots
# from Attempt 1 remain in outputs/clap and plots/clap.
# crc-jeevster (Attempt 3) is whole_clip=True -- ignores CHUNK_SIZE_S, embeds each
# clip as a single unit (see crc_jeevster.md).
MODELS_TO_RUN = ["mert-95m", "melodysim", "crc-jeevster"]

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE_S = 2.0       # seconds per chunk (Attempt 2: matches melody timescale)
CHUNK_OVERLAP = 0.5      # fraction overlap between consecutive chunks (0.5 = 50%)

# ── Storage ───────────────────────────────────────────────────────────────────
# outputs/2s/<model_name>/<split>_<idx>.npz
# Kept under a chunk-size-tagged subdir so this run doesn't collide with / overwrite
# the 10s outputs from Attempt 1 (outputs/<model_name>/...).
OUTPUT_DIR = Path(__file__).parent / "outputs" / "2s"
