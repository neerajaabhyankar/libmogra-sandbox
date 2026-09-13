#!/usr/bin/env bash
# BATCH 11 -- how many clips per performance does the dataset need? ~2.5 h, 4 runs.
#
# Batch 10 draws fresh windows every epoch, so over a run the model sees most of every
# recording -- it answers "does more audio help", not "how many clips should a dataset
# keep". These runs do: each trains on a FIXED set of N evenly spaced 20 s windows per
# performance, reused every epoch, exactly as a dataset with N clips per video would be.
#
#   clips5    N = 5, the Hub's count, but spread over the whole recording
#   clips10   N = 10
#   clips20   N = 20
#   clips40   N = 40 (short recordings give overlapping windows, as a dataset would)
#
# Every run is given the same budget -- about 65k windows seen, i.e. epochs = 65k / (270 N)
# -- so the curve compares data, not training length. Val/test are the Hub clips, as
# everywhere; full_aug (fresh windows, 3 seeds) is the curve's "unlimited" end and
# aug_jitter (the Hub's own 5 clips) is its start.
#
#   bash scripts/run_batch11_clips_per_performance.sh
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

POOL="--arch cqt --batch-size 16 --select-on top1 --tonic normalise --stage 7 \
      --db-head --db-bins 12 --db-lam 0.3 --freq-jitter 2 --train-source full --window-pool"

run clips5   $POOL --windows-per-video 5  --epochs 48 --patience 12
run clips10  $POOL --windows-per-video 10 --epochs 24 --patience 6
run clips20  $POOL --windows-per-video 20 --epochs 12 --patience 3
run clips40  $POOL --windows-per-video 40 --epochs 6  --patience 3

# replication of the one step the curve shows (5 -> 10), paired with aug_seed1/2 and
# full_aug_seed1/2; identical settings to clips10, filter included, so the pairing is clean
run clips10_seed1  $POOL --windows-per-video 10 --epochs 24 --patience 6 --seed 1
run clips10_seed2  $POOL --windows-per-video 10 --epochs 24 --patience 6 --seed 2

echo "== full-window scores"
( cd "$SURVEY" && poetry run python scripts/92_score_full.py clips5 clips10 clips20 clips40 \
    clips10_seed1 clips10_seed2 )
report
