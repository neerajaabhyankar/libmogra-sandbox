#!/usr/bin/env bash
# BATCH 12 -- the two replicated wins together. ~2 h of GPU, then ~12 min of CPU fusions.
#
#   Batch 11 (3 seeds): 10 fixed clips per performance -- +0.038 val over the Hub's 5, and
#                       indistinguishable from the full recordings
#   Batch 9  (3 seeds): mean + std pooling over time -- +0.043 val over the mean
#
# Both are order-free and work through different mechanisms (more views of each
# performance; texture in the summary), so they may add. Three seeds, paired with clips10.
# Each model is then fused with the Essentia and CREPE melody branches, and so is plain
# clips10, so the fused comparison is paired as well.
#
#   nohup bash scripts/run_batch12_stats_clips10.sh > /tmp/batch12.out 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

POOL="--arch cqt --batch-size 16 --select-on top1 --tonic normalise --stage 7 \
      --db-head --db-bins 12 --db-lam 0.3 --freq-jitter 2 --train-source full --window-pool \
      --windows-per-video 10 --epochs 24 --patience 6 --cqt-pool stats"

run clips10_stats        $POOL
run clips10_stats_seed1  $POOL --seed 1
run clips10_stats_seed2  $POOL --seed 2

echo "== full-window scores"
( cd "$SURVEY" && poetry run python scripts/92_score_full.py clips10_stats clips10_stats_seed1 \
    clips10_stats_seed2 )

for dl in clips10 clips10_seed1 clips10_seed2 clips10_stats clips10_stats_seed1 clips10_stats_seed2; do
  once "fuse_${dl}_melody_essentia" scripts/20_fuse_symbolic.py --dl "$dl" --symbolic melody \
       --tracker essentia --device cpu --run-id "fuse_${dl}_melody_essentia"
  once "fuse_${dl}_melody" scripts/20_fuse_symbolic.py --dl "$dl" --symbolic melody \
       --tracker crepe --device cpu --run-id "fuse_${dl}_melody"
done

report
