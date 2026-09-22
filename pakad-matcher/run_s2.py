"""S2 negative control: does a phrase score better in its own raag than where it's merely playable?

For each kept phrase P of raag A, the best candidate cost per clip, over train clips of:
    own      raag A
    twin     raags with exactly A's scale
    legal    other raags whose scale contains every swar of P (P is playable there)
    illegal  a sample of raags missing some swar of P (sanity: should score badly)
    shuffle  up to S2_N_SHUFFLES re-orderings of P, on raag A's clips (does order matter?)

    poetry run python run_s2.py --tag v5 [--raags ...]   # default: all 50 raags' kept phrases
"""

import argparse
import csv
from multiprocessing import Pool

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import contour
import matcher
import phrases
from utils import raagdb


def shuffles(swars, n, rng, tries=2000):
    """Up to n distinct random re-orderings of `swars`, no repeated neighbours, not the original."""
    out = set()
    for _ in range(tries):
        q = tuple(rng.permutation(swars))
        if q != tuple(swars) and all(a != b for a, b in zip(q, q[1:])):
            out.add(q)
        if len(out) == n:
            break
    return sorted(out)


def tasks(p, scales, clips_by_raag, rng):
    own = scales[p.raag]
    need = set(p.swars)
    groups = {"own": clips_by_raag[p.raag], "twin": [], "legal": [], "illegal": []}
    for r, cs in clips_by_raag.items():
        if r == p.raag:
            continue
        groups["legal" if need <= scales[r] else "illegal"] += cs
        if scales[r] == own:
            groups["twin"] += cs
    ill = groups["illegal"]
    groups["illegal"] = list(rng.choice(ill, size=min(C.S2_ILLEGAL_SAMPLE, len(ill)), replace=False))
    out = [("P", g, p.swars, cid) for g, cs in groups.items() for cid in cs]
    for i, q in enumerate(shuffles(p.swars, C.S2_N_SHUFFLES, rng)):
        out += [(f"shuf{i}", "own", q, cid) for cid in groups["own"]]
    return out


def score(job):
    pid, ts = job
    rows = []
    for variant, group, swars, cid in ts:
        cands = matcher.match(contour.contour(cid), swars, top_k=1)
        rows.append((pid, variant, "".join(raagdb.SWAR_NAMES[s] for s in swars), group, cid,
                     round(cands[0].cost if cands else matcher.INF, 3)))
    return rows


def auc(pos, neg):
    """P(pos cost < neg cost), ties count half. Lower cost = better match."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    if not pos.size or not neg.size:
        return np.nan
    return float(np.mean((pos < neg) + 0.5 * (pos == neg)))


def summarise(p, rows):
    by = {}
    for _, variant, _, group, cid, cost in rows:
        by.setdefault((variant, group), []).append((cid, cost))
    get = lambda v, g: np.array([c for _, c in by.get((v, g), [])])
    own, legal, twin = get("P", "own"), get("P", "legal"), get("P", "twin")
    shuf = np.concatenate([get(v, "own") for v, _ in by if v.startswith("shuf")] or [np.array([])])

    def per_video(v, g):
        vid = {}
        for cid, c in by.get((v, g), []):
            k = contour.clips()[cid].video
            vid[k] = min(vid.get(k, np.inf), c)
        return np.array(list(vid.values()))

    thr = np.quantile(legal, C.S2_FPR) if len(legal) else np.nan
    a_null, a_shuf = auc(own, legal), auc(own, shuf)
    return dict(
        phrase_id=p.id, raag=p.raag, phrase=p.text, len=len(p.swars), idf=p.idf,
        n_own=len(own), n_legal_raags=len({contour.clips()[c].raag for c, _ in by.get(("P", "legal"), [])}),
        auc_legal=round(a_null, 3), auc_legal_video=round(auc(per_video("P", "own"), per_video("P", "legal")), 3),
        auc_twin=round(auc(own, twin), 3), auc_shuffle=round(a_shuf, 3),
        hit_at_fpr=round(float(np.mean(own <= thr)), 3) if len(legal) else np.nan,
        med_own=round(float(np.median(own)), 2), med_legal=round(float(np.median(legal)), 2) if len(legal) else np.nan,
        med_illegal=round(float(np.median(get("P", "illegal"))), 2) if len(get("P", "illegal")) else np.nan,
        passes=bool(a_null >= C.S2_GATE_AUC_NULL and a_shuf >= C.S2_GATE_AUC_SHUFFLE),
    )


def main(raags, tag):
    out_dir = C.S2_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.S2_SEED)
    names = sorted({c.raag for c in contour.clips().values()})
    scales = {n: r.scale for n, r in raagdb.dataset_raags(names).items()}
    clips_by_raag = {n: [c for c in contour.cached_ids() if contour.clips()[c].raag == n] for n in names}
    kept = phrases.kept_phrases(raags)
    jobs = [(p.id, tasks(p, scales, clips_by_raag, rng)) for p in kept]
    print(f"{len(kept)} phrases, {sum(len(t) for _, t in jobs)} (variant, clip) scorings")

    rows = []
    with Pool(C.S2_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(score, jobs), 1):
            rows += r
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    with open(out_dir / "scores.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["phrase_id", "variant", "swars", "group", "clip_id", "best_cost"]] + rows)

    by_p = {}
    for r in rows:
        by_p.setdefault(r[0], []).append(r)
    summ = [summarise(p, by_p[p.id]) for p in kept]
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader(); w.writerows(summ)
    print(f"passes gate: {sum(s['passes'] for s in summ)}/{len(summ)}")
    figures(out_dir)


def figures(out_dir):
    import plot
    with open(out_dir / "summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    plot.s2_summary(rows, out_dir / "focus.png", C.FOCUS_RAAGS)
    plot.s2_overview(rows, out_dir / "overview.png", C.FOCUS_RAAGS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raags", nargs="+", default=None)
    ap.add_argument("--tag", required=True, help="run name, e.g. the matcher version")
    a = ap.parse_args()
    main(a.raags, a.tag)
