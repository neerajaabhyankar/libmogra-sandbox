#!/usr/bin/env bash
# BATCH 5 -- seed replication of aug_jitter, the best configuration found. ~2 hours.
#
# Batch 4 measured the seed-to-seed spread of a fixed configuration on the 150 test clips:
# sd 0.058, range 0.107. That is larger than almost every test difference this survey has
# reported, so a single test score is not a result.
#
# aug_jitter scored 0.400 test at seed 0 -- level with motif-classifier's 0.400. These two
# runs make that a mean over three seeds instead of one draw. --seed also re-deals the
# grouped split, so the three together capture both sources of variance.
#
#   bash scripts/run_batch5_seeds.sh
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

AUG="--arch cqt --epochs 40 --patience 10 --batch-size 16 --select-on top1 \
     --tonic normalise --stage 4 --db-head --db-bins 12 --db-lam 0.3 \
     --freq-jitter 2 --gain-jitter 3"

run aug_seed1 $AUG --seed 1
run aug_seed2 $AUG --seed 2

report
