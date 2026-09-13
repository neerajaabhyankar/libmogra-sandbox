#!/usr/bin/env bash
# BATCH 8 -- Essentia's Melodia instead of CREPE for the melody branch. CPU only, ~30 min.
#
# The melody branch is a 120-bin pitch histogram read by a logistic regression, and so far
# every histogram came from CREPE. Essentia's Melodia -- the tracker behind Saraga -- gives
# visibly smoother contours: 0.7 % of frame-to-frame steps over 100 cents against CREPE's
# 5.3 % on the same clip, and it drops frames CREPE keeps (70 % voiced against 93 %).
# Octave errors fold away in a histogram, so smoothness only helps through what it implies
# about *which* frames are counted -- the question is whether it does.
#
# Same three seeds, same splits, same classifier as the CREPE runs, so every comparison is
# paired: melody_only_essentia vs melody_only, and each fusion vs its CREPE twin. Fusion
# scores the DL side on the CPU so it never competes with the GPU queue.
#
#   nohup bash scripts/run_batch8_essentia.sh > /tmp/batch8.out 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/_batch.sh"

echo "== pitch tracks: Essentia over the 1960 Hub clips (resumes; skips what is done)"
( cd "$(dirname "$SURVEY")" && RAAG_CACHE_DIR="$SURVEY/cache/tracks" \
    poetry run python -m utils.extract --tracker essentia ) 2>&1 | grep -v "^\[" | tail -2
echo

once melody_only_essentia        scripts/21_melody_only.py --tracker essentia --seed 0 --run-id melody_only_essentia
once melody_only_essentia_seed1  scripts/21_melody_only.py --tracker essentia --seed 1 --run-id melody_only_essentia_seed1
once melody_only_essentia_seed2  scripts/21_melody_only.py --tracker essentia --seed 2 --run-id melody_only_essentia_seed2

for dl in aug_jitter aug_seed1 aug_seed2; do
  once "fuse_${dl}_melody_essentia" scripts/20_fuse_symbolic.py --dl "$dl" --symbolic melody \
       --tracker essentia --device cpu --run-id "fuse_${dl}_melody_essentia"
done

report
