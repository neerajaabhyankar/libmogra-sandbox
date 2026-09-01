#!/usr/bin/env bash
# BATCH 7 -- Stage 5 [hybrid], the feature-level version. ~2 hours.
#
# Batch 6 fused two *finished* models' probabilities and gained +0.073 test. This asks the
# brief's actual stretch question instead: does the naive melody histogram help when it is
# handed to the network as an input and the two are trained together?
#
# The two are not the same experiment. Fusion lets each family commit to a full 50-way
# opinion and then averages them, so it can only combine finished answers. Concatenation
# lets the classifier head read both representations at once, so it can learn that a
# particular CQT pattern means Bageshree *only when* the histogram shows a weak Ga -- an
# interaction that no weighted sum of two output distributions can express. Fusion is the
# safer bet; this one is the one with the higher ceiling.
#
# hybrid_feat is aug_jitter (the best configuration in the survey) plus --melody, so the
# only difference from a run already on the board is the melody vector. Two more seeds,
# because Batch 4 measured a test sd of 0.058 and this survey has been fooled by a single
# seed three times. hybrid_nodb drops the DB-template head to see whether it is still
# earning its place once the histogram is an input.
#
# The control -- the histogram *alone*, same split, same metrics -- is `melody_only`,
# written by scripts/21_melody_only.py: val 0.430, test 0.347.
#
#   nohup bash scripts/run_batch7_hybrid.sh > /tmp/batch7.out 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

BASE="--arch cqt --epochs 40 --patience 10 --batch-size 16 --select-on top1 \
      --tonic normalise --stage 5 --melody --freq-jitter 2 --gain-jitter 3"
DB="--db-head --db-bins 12 --db-lam 0.3"

run hybrid_feat  $BASE $DB
run hybrid_seed1 $BASE $DB --seed 1
run hybrid_seed2 $BASE $DB --seed 2
run hybrid_nodb  $BASE                  # linear head: is the DB head still needed?

report
