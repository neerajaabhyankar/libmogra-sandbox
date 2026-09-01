#!/usr/bin/env bash
# BATCH 2 -- distilHuBERT, Stages 1 and 2. ~12 hours on an M1. Start it and go to bed.
#
# 20 s clips at ~12 min/epoch, 20 epochs with patience 6. The convolutional feature encoder
# is frozen throughout: benchmarked at 3.1 s/step frozen against 87 s/step unfrozen, which
# is 12 min/epoch against 5.5 hours. Pass --unfreeze-encoder only on a GPU.
#
#   d1    the original notebook's recipe, on v1.1, with video-grouped splits.
#   d2n   the same over tonic-normalised audio.
#   d2c   the same over unmodified audio, tonic supplied by FiLM.
#
# Resumable: each run picks up from its own state.pt, and a finished run is skipped. If your
# laptop sleeps or you interrupt it, just start the script again.
#
#   bash scripts/run_batch2_hubert.sh
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

HUB="--arch hubert --epochs 20 --patience 6 --batch-size 8 --select-on top1"

run d1  $HUB --stage 1
run d2n $HUB --stage 2 --tonic normalise
run d2c $HUB --stage 2 --tonic-mode condition

report
