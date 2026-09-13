#!/usr/bin/env bash
# BATCH 9 -- the CQT trunk: how it summarises time, and its shape. ~7 h, 10 runs.
#
# Every run is aug_jitter (the released configuration) with exactly one thing changed, at
# seed 0, on the Hub clips -- so each is paired with aug_jitter's seed-0 split. A single
# seed resolves about 0.06 on val; anything that gains >= 0.03 gets seeds 1 and 2 next.
#
#   time pooling   the trunk already sees 3.7 s per position; `mean` then throws away the
#                  order *between* positions. tconv and gru keep it; stats and attn are the
#                  controls that are richer but still order-free (models/cqtnet.py)
#   shape          the trunk was chosen by reasoning, never swept: width x0.5 / x2,
#                  depth 3 / 5, and frequency resolution -- the default leaves 18 cells
#                  for 4 octaves (~2.7 semitones each); f36 and f72 keep 36 and 72.
#                  One factor at a time around the default.
#
# --gain-jitter is left off: it never reached the CQT dataset, so aug_jitter's "3" was a no-op.
#
#   nohup bash scripts/run_batch9_cqt_trunk.sh > /tmp/batch9.out 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

AUG="--arch cqt --epochs 40 --patience 10 --batch-size 16 --select-on top1 \
     --tonic normalise --stage 6 --db-head --db-bins 12 --db-lam 0.3 --freq-jitter 2"

run pool_tconv $AUG --cqt-pool tconv
run pool_gru   $AUG --cqt-pool gru
run pool_attn  $AUG --cqt-pool attn
run pool_stats $AUG --cqt-pool stats

run arch_w05   $AUG --cqt-width 0.5
run arch_w2    $AUG --cqt-width 2
run arch_d3    $AUG --cqt-depth 3
run arch_d5    $AUG --cqt-depth 5
run arch_f36   $AUG --cqt-freq-pools 2
run arch_f72   $AUG --cqt-freq-pools 1

# the one change that cleared the >= +0.03 bar at seed 0 -- an order-free control, so the
# replication decides whether "richer summary" is real before anything builds on it
run pool_stats_seed1 $AUG --cqt-pool stats --seed 1
run pool_stats_seed2 $AUG --cqt-pool stats --seed 2

echo "== full-window scores"
( cd "$SURVEY" && poetry run python scripts/92_score_full.py aug_jitter pool_tconv pool_gru \
    pool_attn pool_stats arch_w05 arch_w2 arch_d3 arch_d5 arch_f36 arch_f72 \
    pool_stats_seed1 pool_stats_seed2 )
report
