"""Every constant in pakad-matcher. Scripts import from here; nothing is hard-coded elsewhere."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"

# ---- data (pinned; see CLAUDE.md). Paths themselves come from raag-identifier/utils/config.py
DATASET_REPO_ID = "neerajaabhyankar/hindustani-raag-small"
DATASET_REVISION = "326caef0bc01da44ad46e4d9c65a5146da6bcc5b"  # == utils.config v1.1
SPLIT = "train"

# ---- full recordings (read-only; exploration + annotation context, see CLAUDE.md)
FULLAUDIO_DIR = (HERE / ".." / "raag-identifier" / "hindustani-raag-fullaudios").resolve()
FULLAUDIO_CACHE = CACHE_DIR / "f0_essentia_full.npz"
FULLAUDIO_BLOCK_S = 300.0     # Melodia is run in 5-minute blocks: flat memory on hour-long files

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
MUKHYANGAS_JSON = HERE / "neeraja_mukhyangas.json"   # hand-picked phrases; beats the DB

# first-loop raags (plan.md Q2)
FOCUS_RAAGS = ["Bageshree", "Shree", "PuriyaDhanashri", "Malhar",
               "Malkauns", "DarbariKanada", "Lalit", "Bhoopali"]

# the six raags of the first annotation round (neeraja_mukhyangas.json)
ANNOTATION_RAAGS = ["Bageshree", "DarbariKanada", "Malhar", "PuriyaDhanashri", "Shree",
                    "Bheempalasi"]

# Round 2 (2026-09-24): four raags added for the *test* set only. They are deliberately outside
# ANNOTATION_RAAGS, so no notated (training) chunk can share a recording with them.
TEST_ONLY_RAAGS = ["Des", "TilakKamod", "Multani", "Todi", "KaushikDhwani"]

# Round 3 (2026-09-24): fresh raags for *notation* (training). Chosen away from the test raags,
# and including two audav raags, where the dynamics of a five-swar scale may differ.
NOTATION_RAAGS_R3 = ["Yaman", "Bhairav", "Malkauns", "Bhoopali", "Jog", "Kalawati"]

# ---- matcher (S1). Costs are per frame at the downsampled rate.
# Values marked (tuned) were fitted to the 168 annotations by coordinate ascent on per-phrase
# AUC (tune.py, 2026-09-22); everything else is hand-set. See plan.md S4b.
MATCH = dict(
    free_cents=15.0,          # (tuned, was 30) |pitch - swar| within this costs nothing
    scale_cents=50.0,         # beyond free_cents, cost grows 1 per this many cents
    note_cap=2.0,             # (tuned, was 3) max per-frame note cost
    orn_cost=0.6,             # per-frame cost of an ornament excursion between notes
    transit_cost=0.1,         # per-frame cost of a glide/kan within kan_cents of the neighbours
    gap_cost=0.4,             # per-frame cost of a short unvoiced frame inside a match
    max_gap_s=0.35,           # an unvoiced run longer than this breaks any match
    min_dwell_s=0.07,         # each phrase note must be held at least this long
    step_eps=0.01,            # tiny per-frame cost: prefers the tightest interval
    orn_weight=1.0,           # weight of ornament-time fraction in the final score
    kan_cents=200.0,          # glide/kan within this of the neighbouring notes' range is not ornament
    held_slope=800.0,         # (tuned, was 400) cents/s over held_win_s: slower = "sitting" ...
    held_win_s=0.09,          # slope window; Melodia's 10-cent steps make frame-to-frame slope useless
    held_min_s=0.10,          # ... for at least this long = a held note, not a glide/kan
    note_trim=1.0,            # (tuned, was 0.5) fraction of a note's frames that must fit: at 1.0
                              # a note merely passed through no longer counts as sung
    leap_penalty=1.0,         # per step whose direction/octave contradicts the phrase
    register_penalty=0.5,     # per octave between where the phrase is notated and sung. Tuning
                              # wants 0 -- but the pool it tuned on was *built* with this on, so
                              # it never saw the octave-wrong candidates this removes. Kept.
)
TOP_K = 5                     # candidates kept per (clip, phrase)
CANDIDATE_POOL = 20           # distinct regions re-scored before taking the top-k, minimum ...
CANDIDATES_PER_MIN = 8        # ... and this many per minute of audio: a 30-min recording needs
                              # far more than a 20-s clip, or the re-score never sees the winners
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

# ---- S2: negative control (fixed before looking at results)
S2_DIR = RESULTS_DIR / "s2"          # a run writes to S2_DIR / <tag>
S2_N_SHUFFLES = 5             # distinct re-orderings of each phrase, scored on its own raag
S2_ILLEGAL_SAMPLE = 100       # clips sampled from raags lacking one of the phrase's swars
S2_SEED = 0
S2_WORKERS = 6
S2_GATE_AUC_NULL = 0.70       # own raag vs "legal" raags (all phrase swars in scale), clip level
S2_GATE_AUC_SHUFFLE = 0.60    # phrase vs its shuffles, on own-raag clips
S2_FPR = 0.10                 # hit rate reported at this false-positive rate of the legal null

# ---- S3: annotation (own raag only; see plan.md "what counts as a positive")
S3_DIR = HERE / "annotations"
S3_POOL_PER_PHRASE = 12       # candidates offered per phrase ...
S3_BANDS = (("strong", 0.0, 0.30, 5),    # (name, cost lo, hi, how many) -- absolute cost,
            ("mid",    0.3, 0.80, 4),    # deliberately mixed, so there are "no"s to give
            ("weak",   0.8, 1.50, 3))    # past ~1.5 a note is missing outright: no use asking
S3_MAX_S_PER_NOTE = 0.9       # a 3-swar phrase gets snippets of at most ~2.7 s ...
S3_EXTRA_NOTES = 2            # ... holding at most len(phrase)+2 held notes: no 20-note taans
S3_TOP_PER_CLIP = 3           # candidates considered per clip before sampling
S3_MAX_PER_VIDEO = 2          # no recording dominates a phrase's pool
S3_CONTEXT_S = 0.6            # audio context either side of the candidate
S3_TAIL_S = 0.7               # silence appended, so a sequence of clips is easy to follow
S3_SEED = 7
MATCHER_VERSION = "v7"        # stamped on every label, so labels outlive the matcher

# ---- S3 pools built from full recordings (pool v2)
POOL_VERSION = "v2-full"
POOL_SHORTLIST = 40           # best-by-cost shortlist, then spread over tempo (see pool.py)
POOL_TEMPO_BUCKETS = 3        # slow alap / medium / fast taan renderings all get offered
POOL_PER_PHRASE = 14          # candidates offered per phrase, best-first by cost
POOL_TOP_PER_VIDEO = 3        # no recording dominates a phrase's pool
POOL_PER_VIDEO_SEARCH = 4     # candidates pulled from each recording before ranking
CTX_GAP_S = 0.35              # an unvoiced stretch this long ends the musical "sentence"
CTX_MAX_S = 5.0               # ... but never show more than this either side
CTX_MIN_S = 1.5               # ... nor less than this
APP_PORT = 8765

# ---- S4: features fitted to the annotations
LOCAL_CONTEXT_S = 10.0        # window for "loud/fast *compared with what?*" features
FEATURES = ["pitch_cost", "orn_frac", "gap_frac", "leaps", "register",
            "salience_rel", "salience_min", "tempo_ratio", "held_extra"]

# ---- S5b: notation chunks
CHUNK_DIR = S3_DIR / "chunks"
NOTATIONS = S3_DIR / "notations.jsonl"
CHUNK_ALAP_S = 20.0           # a slow stretch gets 20 s ...
CHUNK_TAAN_S = 15.0           # ... a dense one 15 s: about as much as anyone can hold by ear
CHUNKS_PER_RECORDING = 2
CHUNK_RECORDINGS_PER_RAAG = 2
CHUNK_MIN_VOICED = 0.7        # skip stretches that are mostly silence
# The matcher's costs are tuned for *phrase matching* -- strict intonation, because there a near
# miss is usually not the phrase. Notation is a different question: a gesture that leans on a swar
# is that swar, even 40 cents shy of it. These overrides apply to the notation aligner only.
NOTATE_MATCH = dict(free_cents=35.0, scale_cents=70.0, note_trim=0.4, min_dwell_s=0.05)

# Reading a contour with no phrase to guide it. `onset_cost` is what it costs to declare a new
# note: without it, splitting is free and every glide becomes a run of notes. Fitted on the
# notation corpus (s6.py --sweep), which is the first parameter this project learned from data.
READ_MATCH = dict(NOTATE_MATCH, onset_cost=2.0)   # fitted: s6.py --sweep, 2026-09-24
NOTATE_COVER_WEIGHT = 0.4     # a selection asserts "the sequence is here", so covering it counts
                              # against fitting it; 0 = take the tightest fit, large = cover at any cost
NOTATE_HELD_WEIGHT = 0.2      # ... but the notes should land on the notes: reward alignments whose
                              # note frames sit on held pitch rather than on the way to it. Small,
                              # because a quick dip that only touches a swar is still that swar
NOTATE_SLACK = 0.5            # when aligning a typed sequence in a selected stretch, prefer the
                              # *fullest* alignment among those costing within this of the best:
                              # free ends are for silence and drone at the edges, not an excuse
                              # to explain a 6 s selection with 0.2 s of it
