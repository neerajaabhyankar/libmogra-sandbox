#!/usr/bin/env bash
# BATCH 1 -- the cheap architectures, Stages 1 and 2. ~4 hours total on an M1.
#
# What it answers: does the hand-annotated tonic help a deep model, and by how much, on the
# two architectures where that question costs 45 minutes instead of 4 hours.
#
#   c1           CQT with a fixed fmin -- absolute pitch, no tonic. The control.
#   c2           CQT anchored so bin 0 IS Sa. The structural version of the tonic.
#   c2_shuffled  c2 with tonics permuted between videos. MUST score clearly worse than c2;
#                if it does not, the tonic is not reaching the model and c2 means nothing.
#   r1           jeevster ResNet, the sibling project's best recipe, retrained on v1.1.
#   r2n          the same over tonic-normalised (pitch-shifted) audio.
#   r2c          the same over unmodified audio, with the tonic supplied by FiLM instead.
#
#   bash scripts/run_batch1_cheap.sh
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

CQT="--arch cqt --epochs 40 --patience 10 --batch-size 16 --select-on top1"
RES="--arch resnet1d --epochs 30 --patience 8 --batch-size 16 --unfreeze-blocks 2 --select-on top1"

run c1          $CQT --stage 1
run c2          $CQT --stage 2 --tonic normalise
run c2_shuffled $CQT --stage 2 --tonic normalise --shuffle-tonics
run r1          $RES --stage 1
run r2n         $RES --stage 2 --tonic normalise
run r2c         $RES --stage 2 --tonic-mode condition

report
