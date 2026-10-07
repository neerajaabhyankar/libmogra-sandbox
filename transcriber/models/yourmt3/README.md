# yourmt3

**What:** YourMT3+ (Chang et al., 2024; arXiv 2407.04822), checkpoint "YPTF.MoE+Multi (noPS)"
(562 MB), the Hugging Face space's default. Code and checkpoint vendored in `vendor/` from the
space `mimbres/YourMT3` (**GPL-3.0**; all of `vendor/` is git-ignored). Dependencies via poetry:
`lightning`, `einops`, `deprecated`.

**Getting `vendor/` back** (git-ignored: third-party code, checkpoints, and a YouTube sign-in
token file the space ships in `amt/src/extras/auth2/`):

```bash
poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mimbres/YourMT3', repo_type='space', local_dir='transcriber/models/yourmt3/vendor', allow_patterns=['amt/src/**', 'model_helper.py', 'README.md', 'requirements.txt', 'amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt', 'amt/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt'])"
```

**Outputs:** note events per instrument -- MIDI program (100 = singing voice, 128 = drums), onset,
offset, **integer** MIDI pitch. No pitch bends: everything is on the A440 semitone grid. (Sa sits
within 8 c of that grid for the median recording here; > 25 c for 12%.)

**Lead selection:** singing-voice notes if any; else the pitched program with the most note-time.
Notes are held over their durations on a 10 ms grid. Settings: `config.YOURMT3`.

**Why a worker process (`worker.py`):** its code has top-level `config`, `model` and `utils`
packages that collide with pakad-matcher's `config` and raag-identifier's `utils`. The adapter
streams audio to it over a pipe. Its model file imports `wandb` (a training logger) for one log
table; the worker puts a do-nothing placeholder in its place rather than installing it.

**Cost:** ~22 s to load, then ~8.6 s per 30 s of audio on MPS (M1).

**First check (one notated 30 s range):** it heard singing voice (27 notes), drums (tabla, 44),
piano (probably the harmonium, 27), bass (23) and a few others. Voice track vs Melodia: voiced
63% (Melodia 82%); where both are voiced, median 70 c apart, 47% within 50 c -- one flat pitch per
note, where Melodia follows meend and ornaments.

**Adaptation:** none yet. Idea: keep its voice-note timing, take pitch within each note from a
finer tracker (Basic Pitch contour, CREPE).
