"""S4: fit the annotations. Does anything predict Neeraja's verdict better than the cost does?

    poetry run python s4.py

Honest evaluation: a candidate's fate is predicted by a model that never saw its recording
(grouped by video) and, separately, never saw its phrase (leave-one-phrase-out) -- the second
is the one that says whether this transfers to a phrase we have not annotated.
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import _bootstrap  # noqa: F401
import config as C
import features as F
import fullaudio
import matcher
import mukhyangas


class Item:
    """A labelled candidate, rebuilt from the pool it was drawn from."""

    def __init__(self, rec, item):
        hop = item["hop"]
        path = np.array(item["path"])
        inside = np.flatnonzero(path != -2)
        self.hop = hop
        self.f0 = int(round(item["win_t0"] / hop)) + int(inside[0])
        self.f1 = int(round(item["win_t0"] / hop)) + int(inside[-1])
        self.path = path[inside[0]: inside[-1] + 1]
        self.t0, self.t1 = rec["t0"], rec["t1"]
        self.pitch_cost, self.orn_frac, self.leaps = rec["pitch_cost"], rec["orn_frac"], rec["leaps"]
        self.cost = rec["cost"]


def table():
    labels = {}
    for line in open(C.S3_DIR / "labels.jsonl"):
        r = json.loads(line)
        if r.get("pool") == C.POOL_VERSION:
            labels[(r["phrase_id"], r["index"])] = r
    phrases = {p.id: p for p in mukhyangas.load(only_annotate=False)}

    rows = []
    for (pid, i), rec in sorted(labels.items()):
        if rec["verdict"] == "unsure":
            continue
        p = phrases[pid]
        pool = json.loads((C.S3_DIR / "pool" / f"{p.slug}.json").read_text())
        it = Item(rec, pool["items"][i])
        f = F.candidate_features(rec["video"], it, p.swars, p.octaves)
        rows.append(dict(phrase_id=pid, phrase=p.text, video=rec["video"], index=i,
                         y=int(rec["verdict"] == "yes"), cost=rec["cost"],
                         comment=rec.get("comment", ""), **f))
    return rows


def auc(score, y):
    """P(a yes scores above a no). `score`: higher = more likely to be the phrase."""
    a, b = np.asarray(score)[np.asarray(y) == 1], np.asarray(score)[np.asarray(y) == 0]
    if not len(a) or not len(b):
        return np.nan
    return float(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))


def per_phrase(score, y, pid):
    pid = np.asarray(pid)
    return {p: auc(np.asarray(score)[pid == p], np.asarray(y)[pid == p]) for p in sorted(set(pid))}


def precision_at(score, y, pid, k=3):
    pid, score, y = np.asarray(pid), np.asarray(score), np.asarray(y)
    out = []
    for p in sorted(set(pid)):
        m = pid == p
        top = np.argsort(-score[m])[:k]
        out.append(y[m][top].mean())
    return float(np.mean(out))


def cv_scores(X, y, groups, splitter):
    pred = np.zeros(len(y), float)
    for tr, te in splitter.split(X, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    return pred


def main():
    rows = table()
    y = np.array([r["y"] for r in rows])
    pid = [r["phrase_id"] for r in rows]
    X = np.array([[r[f] for f in C.FEATURES] for r in rows], float)
    cost = np.array([r["cost"] for r in rows])
    print(f"{len(rows)} labels, {y.mean():.2f} yes, {len(set(pid))} phrases, "
          f"{len({r['video'] for r in rows})} recordings\n")

    print("single features, AUC (higher = separates yes from no):")
    print(f"  {'-cost (baseline)':22s} {auc(-cost, y):.3f}   per-phrase mean "
          f"{np.nanmean(list(per_phrase(-cost, y, pid).values())):.3f}")
    for j, f in enumerate(C.FEATURES):
        a = auc(X[:, j], y)
        pp = np.nanmean(list(per_phrase(X[:, j], y, pid).values()))
        print(f"  {f:22s} {a:.3f}   per-phrase mean {pp:.3f}"
              f"{'   (inverted: ' + format(1 - a, '.3f') + ')' if a < 0.5 else ''}")

    print("\nfitted model, cross-validated:")
    for name, splitter, groups in [
            ("unseen recording (GroupKFold by video)", GroupKFold(n_splits=5),
             np.array([r["video"] for r in rows])),
            ("unseen phrase  (leave-one-phrase-out)", LeaveOneGroupOut(), np.array(pid))]:
        pred = cv_scores(X, y, groups, splitter)
        pp = np.nanmean(list(per_phrase(pred, y, pid).values()))
        print(f"  {name:40s} AUC {auc(pred, y):.3f}   per-phrase mean {pp:.3f}"
              f"   P@3 {precision_at(pred, y, pid):.2f}")
    print(f"  {'cost baseline':40s} AUC {auc(-cost, y):.3f}   per-phrase mean "
          f"{np.nanmean(list(per_phrase(-cost, y, pid).values())):.3f}"
          f"   P@3 {precision_at(-cost, y, pid):.2f}")

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    coef = model[-1].coef_[0]
    print("\nfitted on everything -- what the model learned (standardised weights):")
    for f, w in sorted(zip(C.FEATURES, coef), key=lambda t: -abs(t[1])):
        print(f"  {f:22s} {w:+.2f}  {'-> yes' if w > 0 else '-> no'}")

    C.RESULTS_DIR.mkdir(exist_ok=True)
    out = C.RESULTS_DIR / "s4"
    out.mkdir(exist_ok=True)
    import csv
    with open(out / "features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out / 'features.csv'}")


if __name__ == "__main__":
    main()
