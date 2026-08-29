"""Tuning on train (grouped by video), single-shot reporting on test.

The split rule that matters: chunks named `..._[VIDEOID]_chunkN.mp3` come in threes from
one recording. Validating on a clip whose siblings are in the fitting fold measures
recording recall, not raag recognition — so every fold boundary here is a *video*
boundary, and the held-out test split is only ever touched by `--split test`.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from extract import DATA_VERSION
from features import build_features
from represent import Params, build_clips

HERE = Path(__file__).resolve().parent
# v0 results stay exactly where plan.md links them; v1 gets its own subdirectory so the
# two dataset versions never overwrite each other's sweeps.
RESULTS = HERE / "results" / ("" if DATA_VERSION == "v0" else DATA_VERSION)


# ---------------------------------------------------------------- splits


def group_folds(clips, n_folds=5, seed=0):
    """Assign each *video* to a fold, stratifying so every raag appears in every fold."""
    by_raag = defaultdict(list)
    for c in clips:
        by_raag[c.raag].append(c.video)
    rng = np.random.default_rng(seed)
    fold_of_video = {}
    for raag, videos in by_raag.items():
        vids = sorted(set(videos))
        rng.shuffle(vids)
        for i, v in enumerate(vids):
            fold_of_video[v] = i % n_folds
    return fold_of_video


# ---------------------------------------------------------------- metrics


def evaluate(method, feats, top_k=(1, 5), keep_scores=False):
    """Returns a dict of metrics plus per-clip predictions.

    `keep_scores` attaches the raw per-raag score vector to each row — needed by the
    musical-affinity evaluation, which grades the whole ranking rather than just the top-1.
    """
    raags = method.raags
    idx = {r: i for i, r in enumerate(raags)}
    hits = {k: 0 for k in top_k}
    ranks, rows = [], []
    n_empty = 0
    for f in feats:
        if f.n_notes < 2:
            n_empty += 1
        s, k_shift = method.score(f)
        order = np.argsort(-s)
        true_i = idx.get(f.clip.raag)
        rank = int(np.where(order == true_i)[0][0]) + 1 if true_i is not None else len(raags)
        ranks.append(rank)
        for k in top_k:
            hits[k] += rank <= k
        rows.append(
            {
                "clip_id": f.clip.clip_id,
                "true": f.clip.raag,
                "pred": raags[int(order[0])],
                "rank": rank,
                "shift": k_shift,
                **({"scores": [float(x) for x in s]} if keep_scores else {}),
            }
        )
    n = max(len(feats), 1)
    m = {f"top{k}": hits[k] / n for k in top_k}
    m["mrr"] = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0
    m["mean_rank"] = float(np.mean(ranks)) if ranks else 0.0
    m["n_clips"] = len(feats)
    m["n_degenerate"] = n_empty
    return m, rows


def evaluate_by_video(rows, raags=None):
    """Aggregate the chunks of one video into a single verdict — the metric a real user
    would care about, since nobody identifies a raag from 10 seconds in isolation.

    Chunks are pooled by **summing the per-raag score vectors** when they are available.
    A plain majority vote is unusable here: test videos contribute only 2 chunks, so a
    disagreement is a 1-1 tie, and breaking it by `max(set(preds), ...)` made the number
    depend on Python's per-process string hash seed. Summing scores has no ties to break
    and uses the models' confidence rather than throwing it away.
    """
    import re

    by_video = defaultdict(list)
    for r in rows:
        v = re.search(r"\[(.+)\]", r["clip_id"]).group(1)
        by_video[(v, r["true"])].append(r)
    correct = 0
    for (v, true), rs in by_video.items():
        if all("scores" in r for r in rs) and raags is not None:
            pooled = np.sum([np.asarray(r["scores"], dtype=float) for r in rs], axis=0)
            vote = raags[int(np.argmax(pooled))]
        else:  # deterministic fallback: majority, ties broken by name
            preds = [r["pred"] for r in rs]
            vote = max(sorted(set(preds)), key=preds.count)
        correct += vote == true
    return correct / max(len(by_video), 1), len(by_video)


# ---------------------------------------------------------------- method registry


def make_method(name, **kw):
    if name == "m1":
        from methods.m1_exact import ExactPhraseMatcher

        return ExactPhraseMatcher(**kw)
    if name == "m2":
        from methods.m2_ngram import NgramPhraseMatcher

        return NgramPhraseMatcher(**kw)
    if name == "m3":
        from methods.m3_grammar import GrammarMatcher

        return GrammarMatcher(**kw)
    if name == "m4":
        from methods.m4_fusion import TrackerFusion

        return TrackerFusion(**kw)
    if name == "m5":
        from methods.m5_channel import ChannelGrammar

        return ChannelGrammar(**kw)
    if name == "m6":
        from methods.m6_jointtonic import JointTonic

        return JointTonic(**kw)
    if name == "m7":
        from methods.m7_combo import Combo

        return Combo(**kw)
    if name == "m9":
        from methods.m9_tdms import TDMS

        return TDMS(**kw)
    if name == "m9plus":
        from methods.m9_tdms import TDMSPlus

        return TDMSPlus(**kw)
    if name == "m10":
        from methods.m10_register import RegisterMethod

        return RegisterMethod(**kw)
    if name == "m11":
        from methods.m11_histogram import HistogramFingerprint

        return HistogramFingerprint(**kw)
    if name == "m13":
        from methods.m13_ngramlm import NgramLM

        return NgramLM(**kw)
    if name == "m12":
        from methods.m12_dbhist import DBHistogram

        return DBHistogram(**kw)
    if name == "m10plus":
        from methods.m10_register import RegisterPlus

        return RegisterPlus(**kw)
    raise ValueError(name)


# ---------------------------------------------------------------- runner

_FEATURE_KEYS = {"n_min", "n_max", "max_skip", "skip_decay"}


def get_feats(rep_params: Params, split=None, feature_kw=None, _cache={}):
    key = (rep_params, tuple(sorted((feature_kw or {}).items())))
    if key not in _cache:
        clips = build_clips(rep_params)
        _cache[key] = build_features(clips, **(feature_kw or {}))
    feats = _cache[key]
    if split:
        feats = [f for f in feats if f.clip.split == split]
    return feats


def run(method_name, rep_params, method_kw, split="train", feature_kw=None):
    feats = get_feats(rep_params, split=split, feature_kw=feature_kw)
    method = make_method(method_name, **method_kw)
    return evaluate(method, feats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="m3", choices=["m1", "m2", "m3", "m4", "m5", "m6", "m7"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--tracker", default="tony")
    ap.add_argument("--tonic-mode", default="video")
    ap.add_argument("--shift-mode", default="global")
    ap.add_argument("--min-dur", type=float, default=0.0)
    args = ap.parse_args()

    p = Params(tracker=args.tracker, tonic_mode=args.tonic_mode, min_dur=args.min_dur)
    t0 = time.time()
    m, rows = run(args.method, p, {"shift_mode": args.shift_mode}, split=args.split)
    vid_acc, n_vid = evaluate_by_video(rows)
    print(json.dumps(m, indent=1))
    print(f"video-vote top1 {vid_acc:.3f} over {n_vid} videos    ({time.time()-t0:.1f}s)")
