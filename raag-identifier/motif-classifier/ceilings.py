"""What is actually limiting us — transcription, clip length, the DB prior, or dataset size?

Six experiments, all on train with grouped-by-video CV, each isolating one constraint. The
point is to answer "is there hope in the symbolic space at all" with measurements rather
than opinion, and to say which of (a) more data, (b) going back to audio, (c) better
transcription is the lever worth pulling.

    poetry run python ceilings.py
"""

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate import evaluate, group_folds, make_method
from features import ClipFeatures, build_features
from represent import Clip, Params, build_clips
import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.raagdb import dataset_raags
from tune import DEFAULT_REP, cv_score, train_feats

RESULTS = Path(__file__).resolve().parent / "results"
REP = Params(**DEFAULT_REP, collapse_repeats=True)
BEST_M3 = dict(w_arohana=1.0, lam_bi=0.7, lam_uni=0.1, uni_from_scale=0.75,
               w_dur=1.0, w_skip=0.5, shift_mode="none")


# ---------------------------------------------------------------- 1. oracle tonic


def oracle_tonic(feats, method):
    """Upper bound on fixing the tonic: pick each clip's rotation using the true label.

    Cheating on purpose. The gap between this and the honest number is the most that any
    amount of tonic-estimation work could ever buy.
    """
    idx = {r: i for i, r in enumerate(method.raags)}
    hit1 = hit5 = n = 0
    for f in feats:
        if f.n_notes < 2 or f.clip.raag not in idx:
            continue
        t = idx[f.clip.raag]
        scored = [method.score_at(f, k) for k in range(12)]
        best = max(scored, key=lambda s: s[t])  # <- the oracle step
        order = np.argsort(-best)
        rank = int(np.where(order == t)[0][0]) + 1
        hit1 += rank == 1
        hit5 += rank <= 5
        n += 1
    return {"top1": hit1 / max(n, 1), "top5": hit5 / max(n, 1), "n": n}


# ---------------------------------------------------------------- 2. length scaling


def accuracy_by_length(feats, method, edges=(0, 8, 14, 22, 35, 10**9)):
    """Accuracy bucketed by how many notes the clip yielded. If accuracy climbs steeply
    with length, the binding constraint is material per clip, not the size of the corpus."""
    idx = {r: i for i, r in enumerate(method.raags)}
    buckets = defaultdict(lambda: [0, 0])
    for f in feats:
        if f.n_notes < 2 or f.clip.raag not in idx:
            continue
        b = next(i for i in range(len(edges) - 1) if edges[i] <= f.n_notes < edges[i + 1])
        s, _ = method.score(f)
        buckets[b][0] += int(np.argmax(s)) == idx[f.clip.raag]
        buckets[b][1] += 1
    return {
        f"{edges[b]}-{edges[b+1] if edges[b+1] < 10**8 else 'inf'} notes":
            {"top1": c / max(t, 1), "n": t}
        for b, (c, t) in sorted(buckets.items())
    }


# ---------------------------------------------------------------- 3. pooled video


def pool_by_video(clips):
    """Concatenate a video's chunks into one pseudo-clip: 3x the notes, same recording.

    This is the cheapest possible way to get longer sequences without new data, so it
    separates "the clips are too short" from "the dataset is too small".
    """
    by_video = defaultdict(list)
    for c in clips:
        by_video[(c.video, c.raag, c.split)].append(c)
    pooled = []
    for (video, raag, split), cs in by_video.items():
        cs = sorted(cs, key=lambda c: c.clip_id)
        pooled.append(
            Clip(clip_id=f"{raag}/pooled_[{video}].mp3", raag=raag, split=split, video=video,
                 swars=[s for c in cs for s in c.swars],
                 durs=[d for c in cs for d in c.durs],
                 octaves=[o for c in cs for o in c.octaves],
                 tonic_hz=cs[0].tonic_hz)
        )
    return pooled


# ---------------------------------------------------------------- 4-6. data-driven models


class EmpiricalBigram:
    """Per-raag bigram LM estimated from the *transcriptions* of train clips.

    No mukhyanga, no aaroha, no database at all — this is the "what if we had never had
    tanarang.com" baseline, and its learning curve says whether the bottleneck is the
    corpus size.
    """

    def __init__(self, raags, alpha=1.0, lam_uni=0.3, backoff_global=0.2):
        self.raags = list(raags)
        self.alpha, self.lam_uni, self.backoff_global = alpha, lam_uni, backoff_global
        self.fitted = True
        self.shift_mode = "none"
        self.log_bi = np.zeros((len(self.raags), 12, 12))

    def fit(self, feats):
        idx = {r: i for i, r in enumerate(self.raags)}
        R = len(self.raags)
        bi = np.full((R, 12, 12), self.alpha)
        uni = np.full((R, 12), self.alpha)
        for f in feats:
            i = idx.get(f.clip.raag)
            if i is None:
                continue
            s = f.clip.swars
            for a in s:
                uni[i, a] += 1
            for a, b in zip(s, s[1:]):
                bi[i, a, b] += 1
        glob = bi.sum(axis=0)
        glob = glob / glob.sum(axis=1, keepdims=True)
        u = uni / uni.sum(axis=1, keepdims=True)
        p = bi / bi.sum(axis=2, keepdims=True)
        p = (1 - self.lam_uni - self.backoff_global) * p \
            + self.lam_uni * u[:, None, :] + self.backoff_global * glob[None, :, :]
        self.log_bi = np.log(p / p.sum(axis=2, keepdims=True) + 1e-12)
        return self

    def score_at(self, feat, k):
        bg = feat.rot_bigram(k)
        n = bg.sum()
        ll = np.tensordot(self.log_bi, bg, axes=([1, 2], [0, 1]))
        return ll / n if n > 0 else ll

    def score(self, feat):
        return self.score_at(feat, 0), 0


class DbPlusData:
    """The DB grammar and the empirical bigram LM, added in log space."""

    def __init__(self, raags, w_data=1.0, **kw):
        self.db = make_method("m3", **BEST_M3)
        self.data = EmpiricalBigram(raags, **kw)
        self.raags = self.db.raags
        self.fitted = True
        self.shift_mode = "none"
        self.w_data = w_data

    def fit(self, feats):
        self.data.fit(feats)
        return self

    def score_at(self, feat, k):
        return self.db.score_at(feat, k) + self.w_data * self.data.score_at(feat, k)

    def score(self, feat):
        return self.score_at(feat, 0), 0


def grouped_cv(feats, make, n_folds=5, seed=0, frac_videos=1.0):
    """CV where the fitting side can be subsampled *by video* — that is the learning curve."""
    fold = group_folds([f.clip for f in feats], n_folds=n_folds, seed=seed)
    rng = np.random.default_rng(seed)
    tops = []
    for k in range(n_folds):
        held = [f for f in feats if fold[f.clip.video] == k]
        fit = [f for f in feats if fold[f.clip.video] != k]
        if frac_videos < 1.0:
            vids = sorted({f.clip.video for f in fit})
            rng.shuffle(vids)
            keep = set(vids[: max(1, int(len(vids) * frac_videos))])
            fit = [f for f in fit if f.clip.video in keep]
        m = make()
        m.fit(fit)
        tops.append(evaluate(m, held)[0]["top1"])
    return float(np.mean(tops)), float(np.std(tops))


# ---------------------------------------------------------------- driver


def main():
    out = {}
    raags = sorted(dataset_raags())
    feats = train_feats(REP, {})
    m3 = make_method("m3", **BEST_M3)

    print("=== 1. oracle tonic (cheats: picks each clip's rotation using the true label) ===")
    honest = cv_score("m3", REP, BEST_M3)
    orc = oracle_tonic(feats, m3)
    out["oracle_tonic"] = {"honest_top1": honest["top1"], **orc}
    print(f"  honest  top1 {honest['top1']:.3f}   top5 {honest['top5']:.3f}")
    print(f"  oracle  top1 {orc['top1']:.3f}   top5 {orc['top5']:.3f}   "
          f"<- everything tonic estimation could ever buy")

    print("\n=== 2. accuracy vs. how many notes the clip yielded ===")
    out["by_length"] = accuracy_by_length(feats, m3)
    for k, v in out["by_length"].items():
        print(f"  {k:18s} top1 {v['top1']:.3f}  (n={v['n']})")

    print("\n=== 3. pooling a video's 3 chunks into one sequence (3x the notes, no new data) ===")
    pooled = [c for c in pool_by_video(build_clips(REP)) if c.split == "train"]
    pf = build_features(pooled)
    fold = group_folds([f.clip for f in pf])
    per = [evaluate(m3, [f for f in pf if fold[f.clip.video] == k])[0]["top1"] for k in range(5)]
    med = int(np.median([f.n_notes for f in pf]))
    out["pooled"] = {"top1": float(np.mean(per)), "median_notes": med,
                     "unpooled_top1": honest["top1"],
                     "unpooled_median_notes": int(np.median([f.n_notes for f in feats]))}
    print(f"  unpooled  top1 {honest['top1']:.3f}  (median {out['pooled']['unpooled_median_notes']} notes)")
    print(f"  pooled    top1 {out['pooled']['top1']:.3f}  (median {med} notes)")

    print("\n=== 4. data-driven only: bigram LM from transcriptions, no mukhyanga at all ===")
    m, sd = grouped_cv(feats, lambda: EmpiricalBigram(raags))
    out["data_only"] = {"top1": m, "std": sd}
    print(f"  empirical bigram  top1 {m:.3f} ± {sd:.3f}   (DB-only M3 = {honest['top1']:.3f})")

    print("\n=== 5. DB grammar + empirical bigram ===")
    best = None
    for w in (0.25, 0.5, 1.0, 2.0):
        m, sd = grouped_cv(feats, lambda w=w: DbPlusData(raags, w_data=w))
        print(f"  w_data={w:<5} top1 {m:.3f} ± {sd:.3f}")
        if best is None or m > best[1]:
            best = (w, m, sd)
    out["db_plus_data"] = {"w_data": best[0], "top1": best[1], "std": best[2]}

    print("\n=== 6. learning curve: is the empirical model starved of data? ===")
    curve = {}
    n_videos = len({f.clip.video for f in feats})
    for frac in (0.25, 0.5, 0.75, 1.0):
        m, sd = grouped_cv(feats, lambda: EmpiricalBigram(raags), frac_videos=frac)
        mm, _ = grouped_cv(feats, lambda: DbPlusData(raags, w_data=best[0]), frac_videos=frac)
        curve[frac] = {"data_only": m, "db_plus_data": mm,
                       "fit_videos": int(n_videos * 0.8 * frac)}
        print(f"  {int(frac*100):>3}% of fitting videos (~{curve[frac]['fit_videos']:>3}): "
              f"data-only {m:.3f}   DB+data {mm:.3f}")
    out["learning_curve"] = curve

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ceilings.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {RESULTS / 'ceilings.json'}")
    return out


if __name__ == "__main__":
    main()
