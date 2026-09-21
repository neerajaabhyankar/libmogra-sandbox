"""Every constant in pakad-matcher. Scripts import from here; nothing is hard-coded elsewhere."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"

# ---- data (pinned; see CLAUDE.md). Paths themselves come from raag-identifier/utils/config.py
DATASET_REPO_ID = "neerajaabhyankar/hindustani-raag-small"
DATASET_REVISION = "326caef0bc01da44ad46e4d9c65a5146da6bcc5b"  # == utils.config v1.1
SPLIT = "train"

# ---- pitch contour
TRACKER = "essentia"          # utils.extract._essentia (Melodia, 225 fps, 10-cent bins)
F0_CACHE = CACHE_DIR / f"f0_{TRACKER}_v1.1_{SPLIT}.npz"
EXTRACT_WORKERS = 6
DOWNSAMPLE = 4                # 225 fps -> ~56 fps (~18 ms/frame); kan swars are ~60 ms

# ---- phrase catalogue (plan.md Q4)
MIN_PHRASE_LEN = 3            # after collapsing repeats; 2-swar entries dropped
MAX_PHRASE_DF = 9             # drop phrases whose full n-gram occurs in >= 10 DB raags
NGRAM_RANGE = (2, 3)          # sub-n-grams whose IDF defines "idiosyncrasy"
PHRASES_CSV = RESULTS_DIR / "phrases.csv"

# first-loop raags (plan.md Q2)
FOCUS_RAAGS = ["Bageshree", "Shree", "PuriyaDhanashri", "Malhar",
               "Malkauns", "DarbariKanada", "Lalit", "Bhoopali"]

# ---- matcher (S1). Costs are per frame at the downsampled rate.
MATCH = dict(
    free_cents=30.0,          # |pitch - swar| within this costs nothing
    scale_cents=50.0,         # beyond free_cents, cost grows 1 per this many cents
    note_cap=3.0,             # max per-frame note cost
    orn_cost=0.6,             # per-frame cost of an ornament excursion between notes
    transit_cost=0.1,         # per-frame cost of a glide/kan within kan_cents of the neighbours
    gap_cost=0.4,             # per-frame cost of a short unvoiced frame inside a match
    max_gap_s=0.35,           # an unvoiced run longer than this breaks any match
    min_dwell_s=0.07,         # each phrase note must be held at least this long
    step_eps=0.01,            # tiny per-frame cost: prefers the tightest interval
    orn_weight=1.0,           # weight of ornament-time fraction in the final score
    kan_cents=200.0,          # glide/kan within this of the neighbouring notes' range is not ornament
    note_trim=0.5,            # a note is scored on its best-fitting half of frames (andolan-tolerant)
)
TOP_K = 5                     # candidates kept per (clip, phrase)
NMS_IOU = 0.3                 # overlapping candidates above this IoU are suppressed
PLOT_PAD_S = 1.5              # context shown either side of a candidate

# ---- S1 run outputs
S1_DIR = RESULTS_DIR / "s1"
S1_PLOT_TOP = 6               # best candidates plotted per phrase ...
S1_PLOT_MID = 2               # ... plus this many from the median band, for contrast
S1_MAX_PER_VIDEO = 2          # diversity cap on plotted candidates
S1_AUDIO_TOP = 3              # audio snippets written per phrase
S1_COST_BANDS = (0.2, 0.4)    # prevalence reported as #clips with best cost below these

# ---- plot style (dataviz reference palette, light)
STYLE = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0", context="#b9b7af",
    note="#2a78d6", orn="#eb6834", band="#2a78d6", band_alpha=0.06,
    font="DejaVu Sans", row_h=1.9, width=9.0, dpi=130,
)
