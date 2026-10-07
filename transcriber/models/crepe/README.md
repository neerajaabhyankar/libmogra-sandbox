# crepe

**What:** CREPE "full" via `torchcrepe` (weights ship with the pip package; nothing downloaded).
Monophonic: one pitch per 10 ms frame, plus a confidence; voiced where confidence ≥ 0.4.
Code reused from `../raag-identifier/melody-extraction/trackers/crepe_tracker.py`.

**Lead selection:** none -- it follows the strongest periodic sound, so tanpura or harmonium can win
when the voice is soft. That makes it the control for "a learned tracker without lead selection".

**Cost:** ~17 s per 30 s of audio on MPS (M1), first call included.

**First check (2026-10-07, one notated 30 s range):** voiced 85% of frames (Melodia 82%); where
both are voiced, median difference 10 cents, 75% within 50 cents.

**Adaptation:** none yet.
