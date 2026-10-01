"""The notation corpus as training data, and the judgments as validation/test, in one place.

Everything S7 fits comes through `stretches()`; everything it scores comes through `spans()`.
Splits come from `audit.splits()` -- this module never decides what is test.

    stretches()      notated stretches: contour, the swars heard, the recording they came from
    folds(k)         recording-grouped folds over those stretches, for cross-validation
    spans(split)     judged candidate spans for 'validation' or 'test', with their contour

Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import json
from functools import lru_cache

import numpy as np

import _bootstrap  # noqa: F401
import audit
import config as C
import fullaudio
import mukhyangas
from utils import raagdb

SPAN_MARGIN_S = 0.3     # a judged span is re-read with this much context either side


@lru_cache(maxsize=1)
def stretches():
    ch = audit.chunks()
    out = []
    for cid, rec in sorted(audit.notations().items()):
        c = ch[cid]
        if c["video"] in C.BAD_TONIC_VIDEOS:
            continue
        ctr = fullaudio.contour(c["video"])
        for s in rec["segments"]:
            swars, octs = raagdb.parse_phrase(s["swars"].split())
            if not swars:
                continue
            a = int(round((c["t0"] + s["t0"]) / ctr.hop))
            b = int(round((c["t0"] + s["t1"]) / ctr.hop))
            cents = ctr.cents[a:b]
            if len(cents) < 4 or np.isnan(cents).all():
                continue
            out.append(dict(chunk=cid, recording=c["video"], raag=c["raag"], kind=c["kind"],
                            method=s.get("method", "align"), hop=ctr.hop, cents=cents,
                            swars=tuple(swars), octaves=tuple(octs)))
    return out


def folds(k=4, seed=0):
    """Recording-grouped folds: no recording contributes to both sides of a split."""
    recs = sorted({s["recording"] for s in stretches()})
    order = np.random.default_rng(seed).permutation(len(recs))
    fold_of = {recs[i]: j % k for j, i in enumerate(order)}
    return [s for s in stretches()], [fold_of[s["recording"]] for s in stretches()]


@lru_cache(maxsize=2)
def spans(split):
    """Judged spans for one split ('validation' or 'test'), each with its samooha and contour."""
    phrases = {p.id: p for p in mukhyangas.load(only_annotate=False)}
    out = []
    for j in audit.splits()[split]:
        if j["verdict"] not in ("yes", "no"):
            continue
        p = phrases[j["phrase_id"]]
        ctr = fullaudio.contour(j["video"])
        a = max(0, int(round((j["t0"] - SPAN_MARGIN_S) / ctr.hop)))
        b = int(round((j["t1"] + SPAN_MARGIN_S) / ctr.hop))
        out.append(dict(pid=p.id, raag=p.raag, recording=j["video"], y=int(j["verdict"] == "yes"),
                        swars=tuple(p.swars), octaves=tuple(p.octaves), hop=ctr.hop,
                        cents=ctr.cents[a:b]))
    return out


def per_samooha_auc(score, items):
    """Does a 'yes' outrank a 'no' within each samooha? Higher score = more likely the phrase."""
    score = np.asarray(score)
    pid = np.array([it["pid"] for it in items])
    y = np.array([it["y"] for it in items])
    aucs = []
    for p in sorted(set(pid)):
        m = pid == p
        a, b = score[m][y[m] == 1], score[m][y[m] == 0]
        if len(a) and len(b):
            aucs.append(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))
    return float(np.mean(aucs)) if aucs else np.nan


def precision_at(score, items, k):
    score = np.asarray(score)
    pid = np.array([it["pid"] for it in items])
    y = np.array([it["y"] for it in items])
    return float(np.mean([y[pid == p][np.argsort(-score[pid == p])[:k]].mean()
                          for p in sorted(set(pid))]))


if __name__ == "__main__":
    st = stretches()
    print(f"{len(st)} stretches, {sum(len(s['swars']) for s in st)} swars, "
          f"{len({s['recording'] for s in st})} recordings")
    for split in ("validation", "test"):
        sp = spans(split)
        print(f"{split}: {len(sp)} judged spans over {len({s['pid'] for s in sp})} samoohas")
