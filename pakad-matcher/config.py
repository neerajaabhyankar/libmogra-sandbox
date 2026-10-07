"""Every constant in pakad-matcher. Scripts import from here; nothing is hard-coded elsewhere."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
SHARED_RESULTS_DIR = HERE / "results"          # source-independent: phrase catalogue, splits manifest

# ---- pitch source (transcribers/README.md): which pitch track every script reads.
# melodia = Essentia Melodia (results/); any other = a model in ../transcriber, whose numbers go
# to transcribers/<source>/results/. Set per run: PAKAD_PITCH_SOURCE=crepe poetry run python ...
PITCH_SOURCE = os.environ.get("PAKAD_PITCH_SOURCE", "melodia")
TRANSCRIBERS_DIR = HERE / "transcribers"
RESULTS_DIR = (SHARED_RESULTS_DIR if PITCH_SOURCE == "melodia"
               else TRANSCRIBERS_DIR / PITCH_SOURCE / "results")
SEGMENT_PAD_S = 5.0           # context transcribed either side of every range a source must cover

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
PHRASES_CSV = SHARED_RESULTS_DIR / "phrases.csv"
MUKHYANGAS_JSON = HERE / "neeraja_mukhyangas.json"   # hand-picked phrases; beats the DB

# first-loop raags (plan.md Q2)
FOCUS_RAAGS = ["Bageshree", "Shree", "PuriyaDhanashri", "Malhar",
               "Malkauns", "DarbariKanada", "Lalit", "Bhoopali"]

# the six raags of the first annotation round (neeraja_mukhyangas.json)
ANNOTATION_RAAGS = ["Bageshree", "DarbariKanada", "Malhar", "PuriyaDhanashri", "Shree",
                    "Bheempalasi"]

# Raags that are never notated (R4), so the reader cannot have learned them. Their judgments
# test transfer to unseen raags. Round 2 (2026-09-24): Des..KaushikDhwani; round 3 (2026-09-26):
# the rest.
UNNOTATED_RAAGS = ["Des", "TilakKamod", "Multani", "Todi", "KaushikDhwani",
                   "AlhaiyaBilawal", "Chandrakauns", "Bhairavi", "Kedar", "Marwa", "Tilang"]
# R6: judgments in these un-notated raags are VALIDATION, not test -- so validation also holds
# raags the reader never saw (S7 chose badly without them). Fixed before any are judged.
VALIDATION_RAAGS = ["AlhaiyaBilawal", "Tilang"]

# (Which raags were notated in which round is history, not configuration: DATA.md § Inventory.)
UNIDIR_JSON = HERE / "neeraja_unidirectionals.json"   # test2 ground truth
# Recordings whose tonic in the dataset's tonics.csv is wrong (flagged by Neeraja). Excluded
# EVERYWHERE (audit rule R7): notation, judgments, test2 pooling, insight clips. tonics.csv itself
# is not edited here.
BAD_TONIC_VIDEOS = {
    "NMHoLg5PxRM": "Bhairav; Neeraja: 'wrong tonic!!' (2026-09-28); +100 cents fits better",
    "HWukj_DQ8W8": "Multani; Neeraja flagged its insight clip (2026-10-03) and confirmed after "
                   "listening to the whole recording: wrong tonic. tonics.csv says 165.27 Hz (E3 +5c)",
}

# ---- notes from a pitch track (notes.py): one definition for test2, insights and the reader.
# The "next note" rule is Neeraja's (2026-10-03): skip kan, never judge across a breath.
NOTES = dict(
    breath_s=0.25,            # an unvoiced run this long is a breath: no move is judged across it
    min_phrase_s=0.5,         # voiced blips shorter than this between breaths are noise
    kan_max_s=0.08,           # notes shorter than this are kan / pass-through: never "the next note"
)                             # (0.08 s = what the insight train clips preferred, I3)

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
PLOT_PAD_S = 1.5              # context shown either side of a candidate (archive/plot.py)

# ---- S1 run outputs (used only by archive/ scripts)
S1_DIR = RESULTS_DIR / "s1"
S1_PLOT_TOP = 6               # best candidates plotted per phrase ...
S1_PLOT_MID = 2               # ... plus this many from the median band, for contrast
S1_MAX_PER_VIDEO = 2          # diversity cap on plotted candidates
S1_AUDIO_TOP = 3              # audio snippets written per phrase
S1_COST_BANDS = (0.2, 0.4)    # prevalence reported as #clips with best cost below these

# ---- plot style (dataviz reference palette, light; archive/plot.py)
STYLE = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0", context="#b9b7af",
    note="#2a78d6", orn="#eb6834", band="#2a78d6", band_alpha=0.06,
    font="DejaVu Sans", row_h=1.9, width=9.0, dpi=130,
)

# ---- S2: negative control (fixed before looking at results; archive/run_s2.py)
S2_DIR = RESULTS_DIR / "s2"          # a run writes to S2_DIR / <tag>
S2_N_SHUFFLES = 5             # distinct re-orderings of each phrase, scored on its own raag
S2_ILLEGAL_SAMPLE = 100       # clips sampled from raags lacking one of the phrase's swars
S2_SEED = 0
S2_WORKERS = 6
S2_GATE_AUC_NULL = 0.70       # own raag vs "legal" raags (all phrase swars in scale), clip level
S2_GATE_AUC_SHUFFLE = 0.60    # phrase vs its shuffles, on own-raag clips
S2_FPR = 0.10                 # hit rate reported at this false-positive rate of the legal null

# ---- S3: annotation (own raag only). The S3_* pool settings below are archive/s3.py's (pool v1);
# S3_DIR, S3_SEED and MATCHER_VERSION are live
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
POOL_EXTEND_N = 10            # pool.py --extend: deeper candidates appended to an all-yes pool
POOL_EXTEND_SEARCH = 10       # ... searching deeper in each recording
POOL_EXTEND_PER_VIDEO = 6     # ... and letting each recording give more
CTX_GAP_S = 0.35              # an unvoiced stretch this long ends the musical "sentence"
CTX_MAX_S = 5.0               # ... but never show more than this either side
CTX_MIN_S = 1.5               # ... nor less than this
APP_PORT = 8765

# ---- S4: features fitted to the annotations (archive/s4.py, archive/features.py)
LOCAL_CONTEXT_S = 10.0        # window for "loud/fast *compared with what?*" features
FEATURES = ["pitch_cost", "orn_frac", "gap_frac", "leaps", "register",
            "salience_rel", "salience_min", "tempo_ratio", "held_extra"]

# ---- S5b: notation chunks
CHUNK_DIR = S3_DIR / "chunks"
NOTATIONS = S3_DIR / "notations.jsonl"
CHUNK_ALAP_S = 20.0           # a slow stretch gets 20 s ...
CHUNK_TAAN_S = 15.0           # ... a dense one 15 s: about as much as anyone can hold by ear
CHUNK_MADHYA_S = 15.0         # ... and a typical-density one 15 s (the recording's median density)
CHUNK_MADHYA_PER_RAAG = 2     # chunks.py --madhya: recordings per notated raag
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

# ---- S12: learned reader (archive/ctc_reader.py) -- a small network trained on the notation corpus
CTC = dict(
    pc_bins=24, pc_sigma=30.0,        # pitch class as soft bins over the octave (cents)
    hidden=64, layers=2, dropout=0.2,
    epochs=120, batch=4, lr=3e-3, weight_decay=1e-4,
    patience=20,                      # early stop on held-out *recordings* inside the training side
    inner_frac=0.15,
    tempo=(0.8, 1.25), tuning_sd=10.0,  # augmentation: time-stretch range, global detune (cents)
    seeds=(0, 1, 2),                  # an ensemble: log-probs averaged over seeds
    frame_weight=1.0,                 # weight of the per-frame loss on the aligner's note frames
    octaves=1, stride=2,              # classes: 1 = pitch class only, 3 = with octave; frame-rate divisor
    ctc_weight=0.0,                   # weight of the sequence-only (CTC) loss; 0.1 read worse (S12)
)
CTC_DIR = RESULTS_DIR / "ctc_reader"

# ---- Insights (insights/): per-clip aarohi/avarohi swars and nyas swars
INSIGHTS = dict(               # the threshold heuristics (not raag rules); notes come from NOTES
    # aarohi / avarohi
    dir_ratio=2.0,            # aarohi: up >= dir_ratio x down. Values here mirror the frozen
                              # choice (results/insights/choice.json "heuristics", fitted on train +
                              # validation clips, 2026-10-03), which is what actually runs
    dir_min_count=2,          # ... with at least this many moves in the winning direction (I3)
    # nyas = the swar a breath or pause follows
    pause_min_s=0.06,         # a pause: an unvoiced run of at least this (shorter = tracker dropout)
    pause_rel=4.0,            # ... and at least this x the local median note length ("small,
                              # relative to the pace" -- Neeraja) ...
    pause_abs_s=0.25,         # ... or any unvoiced run of at least this, whatever the pace
    pace_window_s=3.0,        # "local": notes within this many seconds
    nyas_min_count=2,         # a nyas swar precedes at least this many pauses ...
    nyas_min_share=0.15,      # ... and at least this share of all pauses
)
INSIGHTS_DIR = RESULTS_DIR / "insights"
INSIGHT_CLIP_S = 30.0         # eyeball clips: one madhya-lay stretch per raag
# Insight clips: one per raag per split, never two splits on one recording (insights/clips.py).
# test and validation never use a notated recording; train may (outside the notated stretches).
INSIGHT_CLIP_RAAGS = {
    "test": ["Yaman", "Bhoopali", "Malkauns", "Chandrakauns", "Des", "TilakKamod", "Multani",
             "Todi", "Kedar", "Marwa", "Tilang", "Madhuvanti", "Sarang", "Bhairavi",
             "AlhaiyaBilawal"],                     # the eyeball set (Bageshree moved to train)
    "validation": ["AheerBhairav", "Durga", "Basant", "KaushikDhwani", "Malkauns", "Charukeshi",
                   "Hindol", "Jog"],
    "train": ["Bageshree", "Bhairav", "Shree", "PuriyaDhanashri", "DarbariKanada", "Bheempalasi",
              "Malhar", "Kalawati", "Yaman", "Bhoopali"],
}
INSIGHT_CLIPS = S3_DIR / "insight_clips.json"       # the registry: append-only, like the pools
INSIGHT_CLIP_DIR = S3_DIR / "insight_clips"
INSIGHT_LABELS = S3_DIR / "insights.jsonl"          # Neeraja's answers, last per clip wins
INSIGHT_VOICE = dict(         # insights/voice.py: loudness above the drone's steady spectrum
    sr=16000, n_fft=1024, band_hz=(150.0, 4000.0),
    floor_pct=20.0,           # a bin's drone level = this percentile of its power over the clip
    floor_mult=2.0,           # energy above this multiple of it counts as voice
    range_db=50.0,            # floor of the scale, below the clip's loud end
)
INSIGHT_DETECT = dict(        # insights/detect.py: learned nyas / direction detectors
    nms_s=1.0,                # two nyas events closer than this: keep the likelier
    dir_min_notes=(0.08, 0.16, 0.3),   # direction counts at these minimum note lengths (all
                                       # >= NOTES.kan_max_s: kan never count, Neeraja 2026-10-03)
    l2_C=0.5,                 # logistic regression: inverse L2 strength (small data)
    thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),   # nyas probability cut, picked on train
)
INSIGHT_DETECT.update(        # notation as proxy direction labels (detect.notation_items)
    proxy_min=4, proxy_one_way=0.85, proxy_both=(0.25, 0.75),
)
