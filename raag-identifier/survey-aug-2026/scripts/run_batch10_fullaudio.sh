#!/usr/bin/env bash
# BATCH 10 -- train on the full recordings instead of the Hub clips. ~6 h, 6 runs.
#
# The Hub keeps ~100 s of each training video out of a median 20 minutes. These runs use
# the *same 270 fit videos* (splits/v1.1_video_splits.csv), drawing fresh 20 s windows
# from the whole recording every epoch; validation and test stay the Hub clips, so every
# number here is directly comparable with every earlier run.
#
#   full_aug x3     aug_jitter with 20 windows/video/epoch -- 4x the Hub's clips per epoch,
#                   and up to 400 distinct windows per video over a run. 20 epochs, since
#                   each is 4x longer. Three seeds: this is the result that could change
#                   the dataset, so it gets the replication up front.
#   full_w5         5 windows/video/epoch on aug_jitter's exact schedule -- the same number
#                   of gradient steps as the Hub run, only the audio differs. Separates
#                   "more distinct audio" from "more steps".
#   full_nofilter   no trim, no loudness filter -- what the gentle filter is worth
#   full_wide       double width: whether 12x the audio supports a bigger trunk
#
# Afterwards every run, and the Hub-trained baselines, are scored on 20 fixed windows of
# every val and test recording (92_score_full.py) -- the neutral comparison.
#
#   nohup bash scripts/run_batch10_fullaudio.sh > /tmp/batch10.out 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

echo "== full-recording CQT cache (resumes; skips what is done)"
( cd "$SURVEY" && poetry run python scripts/04_build_fullaudio_cache.py --workers 3 ) || exit 1
echo

BASE="--arch cqt --batch-size 16 --select-on top1 --tonic normalise --stage 7 \
      --db-head --db-bins 12 --db-lam 0.3 --freq-jitter 2 --train-source full"
FULL="$BASE --epochs 20 --patience 6 --windows-per-video 20"

run full_aug        $FULL
run full_aug_seed1  $FULL --seed 1
run full_aug_seed2  $FULL --seed 2
run full_w5         $BASE --epochs 40 --patience 10 --windows-per-video 5
run full_nofilter   $FULL --trim-seconds 0 --loud-fraction 0
run full_wide       $FULL --cqt-width 2

echo "== full-window scores"
( cd "$SURVEY" && poetry run python scripts/92_score_full.py aug_jitter aug_seed1 aug_seed2 \
    full_aug full_aug_seed1 full_aug_seed2 full_w5 full_nofilter full_wide )
report
