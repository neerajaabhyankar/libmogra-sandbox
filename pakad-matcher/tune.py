"""S4b: how far does *tuning the existing heuristic* get, with no new features?

The labelled spans are fixed; only the cost that ranks them changes. Coordinate ascent over
the re-scoring knobs, objective = mean per-phrase AUC (does a yes outrank a no *within* a
phrase). Reported leave-one-phrase-out, so the number is the gain on a phrase never tuned on.

    poetry run python tune.py
"""

import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import fullaudio
import matcher
import mukhyangas

GRID = {
    "free_cents":       [0.0, 15.0, 30.0, 50.0, 75.0],
    "scale_cents":      [25.0, 50.0, 100.0, 200.0],
    "note_cap":         [1.0, 2.0, 3.0],
    "note_trim":        [0.25, 0.5, 0.75, 1.0],
    "orn_weight":       [0.0, 0.5, 1.0, 2.0, 4.0],
    "kan_cents":        [50.0, 100.0, 200.0, 350.0],
    "held_min_s":       [0.06, 0.10, 0.20, 0.40],
    "held_slope":       [200.0, 400.0, 800.0],
    "leap_penalty":     [0.0, 0.5, 1.0, 2.0],
    "register_penalty": [0.0, 0.25, 0.5, 1.0, 2.0],
}


def spans():
    """Each labelled candidate as (phrase_id, y, cents, kinds, swars, octaves, hop)."""
    phrases = {p.id: p for p in mukhyangas.load(only_annotate=False)}
    seen = {}
    for line in open(C.S3_DIR / "labels.jsonl"):
        r = json.loads(line)
        if r.get("pool") == C.POOL_VERSION:
            seen[(r["phrase_id"], r["index"])] = r
    out = []
    for (pid, i), r in sorted(seen.items()):
        if r["verdict"] == "unsure":
            continue
        p = phrases[pid]
        item = json.loads((C.S3_DIR / "pool" / f"{p.slug}.json").read_text())["items"][i]
        path = np.array(item["path"])
        inside = np.flatnonzero(path != -2)
        cents = np.array([np.nan if c is None else c for c in item["cents"]])
        out.append(dict(pid=pid, y=int(r["verdict"] == "yes"), video=r["video"],
                        cents=cents[inside[0]: inside[-1] + 1],
                        kinds=path[inside[0]: inside[-1] + 1],
                        swars=p.swars, octaves=p.octaves, hop=item["hop"]))
    return out


def costs(items, params):
    p = {**C.MATCH, **params}
    return np.array([matcher.score_path(it["cents"], it["kinds"], it["swars"], it["octaves"],
                                        it["hop"], p)[-1] for it in items])


def per_phrase_auc(cost, items, only=None):
    pid = np.array([it["pid"] for it in items])
    y = np.array([it["y"] for it in items])
    aucs = []
    for p in sorted(set(pid)) if only is None else [only]:
        m = pid == p
        a, b = -cost[m][y[m] == 1], -cost[m][y[m] == 0]
        if len(a) and len(b):
            aucs.append(float(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :]))))
    return float(np.mean(aucs)) if aucs else np.nan


def coordinate_ascent(items, start=None, rounds=3, exclude=None):
    keep = [it for it in items if exclude is None or it["pid"] != exclude]
    best = dict(start or {})
    score = per_phrase_auc(costs(keep, best), keep)
    for _ in range(rounds):
        improved = False
        for k, values in GRID.items():
            for v in values:
                if best.get(k, C.MATCH[k]) == v:
                    continue
                trial = {**best, k: v}
                s = per_phrase_auc(costs(keep, trial), keep)
                if s > score + 1e-6:
                    best, score, improved = trial, s, True
        if not improved:
            break
    return best, score


def main():
    items = spans()
    base = per_phrase_auc(costs(items, {}), items)
    print(f"{len(items)} labelled spans, {len({it['pid'] for it in items})} phrases")
    print(f"as shipped:           per-phrase AUC {base:.3f}")

    best, fit = coordinate_ascent(items)
    print(f"tuned on everything:  per-phrase AUC {fit:.3f}   (optimistic -- it saw every phrase)")
    print("  " + ", ".join(f"{k}={v}" for k, v in best.items() if v != C.MATCH[k]))

    held = []
    for pid in sorted({it["pid"] for it in items}):
        pars, _ = coordinate_ascent(items, exclude=pid)
        held.append(per_phrase_auc(costs(items, pars), items, only=pid))
    print(f"leave-one-phrase-out: per-phrase AUC {np.nanmean(held):.3f}   "
          f"(tuned without that phrase, scored on it)")
    for pid, h in zip(sorted({it["pid"] for it in items}), held):
        print(f"    {pid:22s} {h:.2f}")


if __name__ == "__main__":
    main()
