"""Twin-restricted accuracy under proper grouped CV, plus the confusion inside one group.

`scale_twins.py` fits on all of train and scores the same clips, which flatters any method
that learns templates. This refits per fold like the real harness does, then reports
accuracy restricted to raags that share a scale with another raag — the only place where
pitch-set information is useless and phrase/emphasis information has to do the work.
"""

import argparse
from collections import Counter, defaultdict

import numpy as np

from evaluate import make_method, group_folds
import _bootstrap  # noqa: F401  (puts raag-identifier/ on sys.path)
from utils.raagdb import dataset_raags
from represent import Params
from tune import train_feats

BASE_REP = dict(tracker="tony", note_source="hmm", tonic_mode="true", tonic_refine=True,
                min_dur=0.0, max_cents_dev=2400.0, collapse_repeats=True)
HIST = dict(n_bins=120, source="frames", tracker="crepe", metric="chi2",
            smooth=1.0, power=0.5, tonic_mode="true")
METHODS = {
    "m3":  ("m3",  dict(shift_mode="none", w_arohana=1.0, symmetric=False, lam_bi=0.7,
                        lam_uni=0.1, uni_from_scale=0.75, nyas_boost=1.0, w_dur=1.0,
                        dur_weighted=False, w_skip=0.5), ()),
    "m9":  ("m9",  dict(tracker="crepe", n_bins=80, tau=0.3, smooth=1.0,
                        tonic_mode="true", metric="chi2"), ()),
    "m11": ("m11", dict(HIST), ()),
    "m12": ("m12", dict(HIST, lam=0.3), ()),
}


def twin_groups():
    by_scale = defaultdict(list)
    for name, r in dataset_raags().items():
        by_scale[frozenset(r.scale)].append(name)
    return {k: sorted(v) for k, v in by_scale.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--group", nargs="+", default=["Bageshree", "Bheempalasi", "Kafi"])
    args = ap.parse_args()

    twin_of = {}
    for members in twin_groups().values():
        for m in members:
            twin_of[m] = set(members) - {m}

    rep = Params(**BASE_REP)
    hdr = (f"{'method':<7} {'all top1':>9} {'twin top1':>10} {'->twin':>8} "
           f"{'non-twin top1':>14}")
    print(hdr); print("-" * len(hdr))
    conf = {}
    for name in args.methods:
        real, kw, extra = METHODS[name]
        feats = train_feats(rep, {}, extra)
        fold = group_folds([f.clip for f in feats])
        hit = tot = th = tt = t2t = nh = nt = 0
        cm = Counter()
        for k in sorted(set(fold.values())):
            sub = [f for f in feats if fold[f.clip.video] == k]
            m = make_method(real, **kw)
            if m.fitted:
                m.fit([f for f in feats if fold[f.clip.video] != k])
            for f in sub:
                p, t = m.predict(f), f.clip.raag
                hit += p == t; tot += 1
                if t in twin_of:
                    tt += 1; th += p == t; t2t += p in twin_of[t]
                else:
                    nt += 1; nh += p == t
                if t in args.group:
                    cm[(t, p)] += 1
        conf[name] = cm
        print(f"{name:<7} {hit/max(tot,1):9.3f} {th/max(tt,1):10.3f} "
              f"{t2t/max(tt,1):8.3f} {nh/max(nt,1):14.3f}")

    print(f"\nInside {' / '.join(args.group)} (rows = truth, CV predictions):")
    for name in args.methods:
        cm = conf[name]
        print(f"\n  {name}")
        for t in args.group:
            n = sum(v for (tt_, _), v in cm.items() if tt_ == t)
            inside = {p: v for (tt_, p), v in cm.items() if tt_ == t and p in args.group}
            other = n - sum(inside.values())
            cells = "  ".join(f"{p[:11]}:{inside.get(p,0):>3}" for p in args.group)
            print(f"    {t:<12} n={n:<4} {cells}   elsewhere:{other:>3}")


if __name__ == "__main__":
    main()
