#!/usr/bin/env bash
# BATCH 3 -- source separation (Stage 3) and the libmogra DB as a prior (Stage 4).
# ~4 hours on an M1, including a one-time ~20 min cache build.
#
# Everything here sits on top of Stage 2's winner: the tonic, at the audio/representation
# level. Batch 1 settled that (c2 0.302 vs c1 0.111; r2n 0.287 vs r1 0.146), and settled
# that FiLM conditioning is not the route, so no run below uses --tonic-mode condition.
#
# Stage 3 -- does removing the tabla help?
#   c3   CQT, Sa-anchored, over the HPSS melody stem.
#   r3   jeevster ResNet, normalised audio, over the HPSS melody stem.
#        Prior: motif-classifier tried exactly this for the symbolic methods and every
#        method that saw separated audio got *worse* (M9 -0.028, M12 -0.012, M14 -0.026),
#        the reading being that HPSS smooths away meend and gamak along with the tabla.
#        These two runs ask whether a model that sees the spectrogram rather than a pitch
#        track is hurt the same way. A negative result here is a real result.
#
# Stage 4 -- three different ways to use the database, cheapest first.
#   c4g  graded label smoothing: move 0.3 of the one-hot onto musically adjacent raags.
#        Touches the loss only, so it is the cleanest read on "does the DB help at all".
#   c4a  auxiliary head predicting swar occupancy, weighted 0.3. The DB as a second task.
#   c4h  the DB-template head: predict a 12-bin swar profile and score it against the
#        libmogra templates by chi-square. This is M12's mechanism, learned end to end,
#        and it is the reason the C architecture was specified with a 12-bin feature space.
#   r4g  graded label smoothing on the ResNet, to check the loss-level result is not
#        specific to the CQT architecture.
#
#   bash scripts/run_batch3_sep_db.sh
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

# ---- one-time: the HPSS caches (waveform + Sa-anchored CQT). Skips what it already has.
echo "== CACHE hpss (one time, ~20 min; skips clips already built)"
( cd "$SURVEY" && poetry run python scripts/00_build_cache.py --separate hpss ) || {
    echo "!! HPSS cache build failed -- Stage 3 runs will fail; Stage 4 is unaffected"; }
echo

CQT="--arch cqt      --epochs 40 --patience 10 --batch-size 16 --select-on top1"
RES="--arch resnet1d --epochs 30 --patience 8  --batch-size 16 --select-on top1 --unfreeze-blocks 2"

# ---- Stage 3: source separation
run c3  $CQT --stage 3 --tonic normalise --separate hpss
run r3  $RES --stage 3 --tonic normalise --separate hpss

# ---- Stage 4: the database as a prior
run c4g $CQT --stage 4 --tonic normalise --graded-alpha 0.3
run c4a $CQT --stage 4 --tonic normalise --aux-weight 0.3
run c4h $CQT --stage 4 --tonic normalise --db-head --db-lam 0.3 --db-bins 12
run r4g $RES --stage 4 --tonic normalise --graded-alpha 0.3

report
