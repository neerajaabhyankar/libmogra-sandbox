"""Shared machinery for the survey-aug-2026 DL experiments.

Nothing here is experiment-specific. If two scripts need it, it lives in this package;
if one script needs it, it stays in that script.

    common.paths      where things are, and the sibling folders we import from
    common.data       the clip index and video-grouped splits
    common.tonic      Sa arithmetic: cents, octave folding, resample ratios
    common.audio      decode + cache + tonic-normalise + separate + CQT
    common.dbprior    libmogra's raag database, as targets/templates a network can use
    common.metrics    top-1/5, MRR, macro-F1, video-vote, and the musical grading
    common.trainer    one training loop, shared by all three architectures
    common.report     result.json -> tables and confusion matrices
"""
