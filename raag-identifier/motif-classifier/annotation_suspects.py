"""Which tonic annotations look wrong?

Two independent facts say the annotated Sa is not always right: oracle-rotation still
buys +0.077 over it, and 15.5 % of M9+'s errors are near-exact rotations of the true
scale. This locates the specific recordings responsible.

Method: for every clip, score the *true* raag at all 12 rotations of the annotated
tonic and note which rotation the evidence prefers. One clip preferring a rotation means
little — clips are short and the grammar is noisy. What matters is **agreement across
chunks of the same recording**, because the annotation is per-recording: if five chunks
of one video independently prefer the same non-zero rotation, the annotation for that
video is off by that many semitones.

Two scorers are run because they fail differently and agreement between them is much
stronger evidence than either alone:

  * **M3** — smoothed phrase-bigram grammar over quantized swars, prescriptive, fits
    nothing. Sees phrase shape.
  * **M9** — cosine against a time-delayed melody surface, learned from train clips,
    no database. Sees continuous contour.

M9's templates are fit on train, so for train clips it has seen the clip itself; that
inflates its confidence but not its *rotation* preference, which is what is read here.
Ranks are used rather than raw scores so the two are comparable.

    poetry run python annotation_suspects.py

Writes possible-annotation-errors.txt.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate import make_method
from raagdb import dataset_raags
from extract import DATA_VERSION, list_clips
from represent import Params, _load
from tune import train_feats, split_feats

HERE = Path(__file__).resolve().parent

M3_KW = dict(shift_mode="none", w_arohana=1.0, symmetric=False, lam_bi=0.7,
             lam_uni=0.1, uni_from_scale=0.75, nyas_boost=1.0, w_dur=1.0,
             dur_weighted=False, w_skip=0.5)
M9_KW = dict(tracker="crepe", n_bins=60, tau=0.3, smooth=1.0, tonic_mode="true")
BASE_REP = dict(tracker="tony", note_source="hmm", tonic_mode="true", tonic_refine=True,
                min_dur=0.0, max_cents_dev=2400.0, collapse_repeats=True)


def rotation_evidence(feats, method):
    """Per clip: rank of the true raag at each of the 12 rotations of the annotated Sa."""
    idx = {r: i for i, r in enumerate(method.raags)}
    out = []
    for f in feats:
        if f.n_notes < 2 or f.clip.raag not in idx:
            continue
        t = idx[f.clip.raag]
        ranks = []
        for k in range(12):
            s = method.score_at(f, k)
            ranks.append(int(np.sum(s > s[t])) + 1)  # rank of the true raag, 1 = best
        out.append({"clip": f.clip.clip_id, "video": f.clip.video, "raag": f.clip.raag,
                    "split": f.clip.split, "ranks": np.array(ranks)})
    return out


def by_video(rows):
    """Collapse clips to recordings: which rotation do this video's chunks prefer?"""
    g = defaultdict(list)
    for r in rows:
        g[r["video"]].append(r)
    out = {}
    for video, rs in g.items():
        R = np.stack([r["ranks"] for r in rs])          # (n_chunks, 12)
        per_chunk_best = R.argmin(axis=1)
        mean_rank = R.mean(axis=0)
        best_k = int(mean_rank.argmin())
        out[video] = {
            "raag": rs[0]["raag"], "split": rs[0]["split"], "n_chunks": len(rs),
            "best_k": best_k,
            "rank_at_0": float(mean_rank[0]),
            "rank_at_best": float(mean_rank[best_k]),
            "gain": float(mean_rank[0] - mean_rank[best_k]),
            # how many chunks independently pick that same rotation
            "agree": int(np.sum(per_chunk_best == best_k)),
            "chunk_choices": Counter(int(k) for k in per_chunk_best),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-agree", type=float, default=0.6,
                    help="fraction of a video's chunks that must prefer the same rotation")
    ap.add_argument("--min-gain", type=float, default=3.0,
                    help="mean rank places the rotation must recover")
    ap.add_argument("--out", default="possible-annotation-errors.txt")
    args = ap.parse_args()

    tonics = {c["video"]: c["true_tonic_hz"] for c in list_clips()}
    scales = {n: set(r.scale) for n, r in dataset_raags().items()}
    per_scorer = {}
    for name, kw, extra in (("m3", M3_KW, ()), ("m9", M9_KW, ())):
        rep = Params(**BASE_REP)
        train = train_feats(rep, {}, extra)
        test = split_feats(rep, {}, "test", extra)
        m = make_method(name, **kw)
        if m.fitted:
            m.fit(train)  # train only; test clips stay unseen
        per_scorer[name] = by_video(rotation_evidence(list(train) + list(test), m))
        print(f"{name}: {len(per_scorer[name])} videos scored", flush=True)

    videos = sorted(per_scorer["m3"])
    rows = []
    for v in videos:
        a, b = per_scorer["m3"][v], per_scorer["m9"].get(v)
        if b is None:
            continue
        both = a["best_k"] == b["best_k"] and a["best_k"] != 0
        frac_a = a["agree"] / a["n_chunks"]
        frac_b = b["agree"] / b["n_chunks"]
        flagged = (both and min(frac_a, frac_b) >= args.min_agree
                   and min(a["gain"], b["gain"]) >= args.min_gain)
        rows.append({"video": v, "m3": a, "m9": b, "both": both,
                     "frac_a": frac_a, "frac_b": frac_b, "flagged": flagged,
                     "score": min(a["gain"], b["gain"]) if both else -1.0})
    rows.sort(key=lambda r: -r["score"])

    flagged = [r for r in rows if r["flagged"]]
    agreed = [r for r in rows if r["both"]]
    lines = [
        "# Possible tonic-annotation errors",
        "",
        f"dataset: hindustani-raag-small {DATA_VERSION} · {len(rows)} recordings scored",
        "",
        "Each recording's chunks were scored against the TRUE raag at all 12 rotations of",
        "the annotated Sa. A recording is listed when two independent scorers (M3, a",
        "prescriptive phrase grammar; M9, a learned melody surface) agree on the SAME",
        "nonzero rotation. Rotating a clip by `k` is defined as lowering its tonic by",
        "`k`, so **the annotation looks `k` semitones too SHARP** and the evidence",
        "prefers a true tonic of `annotated x 2^(-k/12)` — the `true?` column.",
        "",
        f"* {len(agreed)} recordings where both scorers agree on a nonzero rotation",
        f"* {len(flagged)} of those also clear the thresholds "
        f"(>={args.min_agree:.0%} of chunks agree, >={args.min_gain:.0f} mean rank places recovered)",
        "",
        "STRONG — recheck these first",
        "",
        f"{'video':<14} {'raag':<17} {'k':>3} {'annotated':>10} {'true?':>8} {'sym':>5} "
        f"{'chunks':>7} {'M3 agree':>9} {'M3 rank':>13} {'M9 agree':>9} {'M9 rank':>13}",
    ]
    def fmt(r):
        v, a, b = r["video"], r["m3"], r["m9"]
        k = a["best_k"]
        t = tonics.get(v, float("nan"))
        sc = scales.get(a["raag"], set())
        rot = {(x + k) % 12 for x in sc}
        sym = len(sc & rot) / max(len(sc | rot), 1)
        return (f"{v:<14} {a['raag']:<17} {k:>3} {t:>10.1f} {t * 2 ** (-k / 12):>8.1f} "
                f"{sym:>5.2f} "
                f"{a['n_chunks']:>7} {a['agree']}/{a['n_chunks']:<7} "
                f"{a['rank_at_0']:>5.1f}->{a['rank_at_best']:<6.1f} "
                f"{b['agree']}/{b['n_chunks']:<7} "
                f"{b['rank_at_0']:>5.1f}->{b['rank_at_best']:<6.1f}")
    lines += [fmt(r) for r in flagged] or ["  (none)"]
    lines += ["", "WEAKER — both scorers agree on the rotation but below threshold", ""]
    lines += [fmt(r) for r in agreed if not r["flagged"]] or ["  (none)"]
    lines += [
        "",
        "Notes",
        "",
        "* k=7 means the annotation sits a fifth ABOVE the evidence's Sa — i.e. Pa was",
        "  marked as Sa, the commonest way to mis-hear a tonic. k=5 is the fourth-below",
        "  equivalent.",
        "* `sym` is Jaccard(scale, scale rotated by k) for the true raag. A high value",
        "  means the raag's own scale nearly maps onto itself under this rotation, so the",
        "  evidence is weak by construction — pentatonic raags do this. Treat sym>=0.6",
        "  as suspect even when both scorers agree.",
        "* A recording can appear here for reasons other than a wrong annotation: a",
        "  performance that modulates, a chunk that is mostly tabla or applause, or a raag",
        "  whose grammar genuinely fits a rotation of itself. Listen before editing.",
        "* Absence from this list is not proof an annotation is right — it means these two",
        "  scorers did not agree it was wrong.",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
