#!/usr/bin/env bash
# BATCH 4 -- Stage 4 [DB prior] follow-ups, plus the rigour that Batches 1-3 skipped.
# ~4 hours on an M1. Stage 5 [hybrid] is a separate script; see scripts/20_fuse_symbolic.py.
#
# Batch 3 found the DB-template head worth +0.115 (c4h, 0.417 val / 0.387 test) -- the
# biggest single win in the survey. But c4h is the FIRST point sampled, not a tuned one:
# --db-bins ran only at 12 of {12,36,144}, --db-lam only at 0.3, and the ablation that
# separates "the database helps" from "a template-shaped head helps" was never run.
#
#   dbprior_lam0     --db-lam 0. Learned templates, identical architecture. THE missing
#                    ablation: without it, +0.115 cannot be attributed to the database.
#   dbprior_36bins   3 bins/semitone (~33 cents). Stage 3 showed sub-semitone movement is
#                    signal; a 12-bin profile quantises meend and gamak away.
#   dbprior_144bins  the CQT's own resolution, folded -- matches dbprior.pitch_template.
#   dbprior_frozen   --db-lam 1 --db-freeze-templates: 50 scalar biases as the only
#                    raag-specific parameters. The extreme end of "let the DB classify".
#
# Rigour. Across all 16 runs so far, seed / folds / jitter never varied once.
#   aug_jitter       --freq-jitter 2 --gain-jitter 3. Sub-semitone pitch jitter was written
#                    for this architecture and has run at 0 every time. ~36 clips/class.
#   seed1, seed2     c4h again at other seeds. --seed moves the grouped split too, so the
#                    three TEST scores together are the honest error bar on 0.387.
#   cv5              --folds 5: out-of-fold over all 1810 clips instead of judging on 460.
#                    ~100 min, and it is what the headline number deserves.
#
#   bash scripts/run_batch4_dbprior.sh
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

# c4h's recipe, which every run below is a variation of
BASE="--arch cqt --epochs 40 --patience 10 --batch-size 16 --select-on top1 --tonic normalise"
DBH="--db-head --db-bins 12 --db-lam 0.3"

# ---- Stage 4 [DB prior]: what is the head actually doing?
run dbprior_lam0    $BASE --stage 4 --db-head --db-bins 12  --db-lam 0.0
run dbprior_36bins  $BASE --stage 4 --db-head --db-bins 36  --db-lam 0.3
run dbprior_144bins $BASE --stage 4 --db-head --db-bins 144 --db-lam 0.3
run dbprior_frozen  $BASE --stage 4 --db-head --db-bins 12  --db-lam 1.0 --db-freeze-templates

# ---- rigour on the winning configuration
run aug_jitter      $BASE --stage 4 $DBH --freq-jitter 2 --gain-jitter 3
run seed1           $BASE --stage 4 $DBH --seed 1
run seed2           $BASE --stage 4 $DBH --seed 2
run cv5             $BASE --stage 4 $DBH --folds 5

report
