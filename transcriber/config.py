"""Every constant of the transcriber package."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"                       # cache/<model>/<key>.npz -- not committed
RAAG_IDENTIFIER = (HERE / ".." / "raag-identifier").resolve()
MELODY_EXTRACTION = RAAG_IDENTIFIER / "melody-extraction"
SOURCE_SEPARATION = RAAG_IDENTIFIER / "source-separation"

DEVICE = "mps"                                   # torch device for models that use torch

CREPE = dict(model="full", sr=16000, hop=160, voiced_conf=0.4)   # fmin 50, fmax 2000 Hz: set in crepe_tracker

# Basic Pitch: lead = a continuity-favouring path through its pitch-salience map (3 bins/semitone)
BASIC_PITCH = dict(fmin_hz=60.0, fmax_hz=1100.0,    # search range: low male voice .. high flute
                   voiced_salience=0.3,             # path frames below this salience are unvoiced
                   jump_cost=0.15,                  # log-salience cost per bin of pitch jump
                   jump_cap_bins=12,                # jumps beyond 4 semitones cost the same
                   bin_offset=-1.0,                 # pure tones 110-440 Hz peak one bin (33 c) high
                                                    # (measured 2026-10-07; models/basic_pitch/README)
                   batch=32)                        # windows per model call

# YourMT3 family (vendored from the Hugging Face space mimbres/YourMT3, GPL-3.0). Checkpoint
# arguments as the space's app.py, fp32 off-GPU. One adapter, one worker; a model subfolder each.
YOURMT3 = dict(
    vendor=HERE / "models" / "yourmt3" / "vendor",
    checkpoints={
        "yourmt3": ["mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
                    "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26",
                    "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8",
                    "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1", "-ac", "spec",
                    "-hop", "300", "-atc", "1", "-pr", "32"],    # "YPTF.MoE+Multi (noPS)"
        "ymt3plus": ["notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt",
                     "-p", "2024", "-pr", "32"],                  # "YMT3+": MT3's architecture
    },
    voice_programs=(100,),        # GM_INSTR_CLASS_PLUS: 100 = singing voice (melody)
    batch=8,
    hop_s=0.01)                   # grid of the held-note track the notes become

# Adaptation head (adapt.py) and its feature blocks (features.py): one 10 ms grid, 20-cent bins
ADAPT = dict(hop_s=0.01, fmin_hz=50.0, n_bins=300, cents_per_bin=20,   # 50 Hz .. 1600 Hz
             cqt_bins=180,                    # 5 octaves at 3 bins per semitone from 50 Hz
             width=128, layers=4, kernel=5,   # head: per-block projection, dilated convolutions
             label_sigma_bins=1.25,           # soft pitch labels (25 cents)
             crop=400, batch=32, steps=1500, lr=1e-3, seed=0,
             voiced_threshold=0.5)            # P(silent) below 1 - this = voiced
