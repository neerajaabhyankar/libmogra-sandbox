"""Raags that share a scale can only be told apart by emphasis — does M10 do it?

M10's premise is that `vaadi`/`samvaadi` carry the signal for pairs whose swar *sets* are
identical, and that the annotated tonic is what makes that signal readable. This checks the
premise directly instead of trusting the aggregate: find every group of raags sharing a
scale, then measure, on those clips only, how often each method lands on a scale-twin
rather than the truth.

    poetry run python scale_twins.py --methods m3 m10
"""

import argparse
from collections import defaultdict

import numpy as np

from evaluate import make_method, group_folds
from raagdb import dataset_raags
from represent import Params
from tune import train_feats

BASE_REP = {"tracker": "tony", "note_source": "hmm", "tonic_refine": True,
            "min_dur": 0.0, "max_cents_dev": 2400.0, "collapse_repeats": True}


def twin_groups():
    by_scale = defaultdict(list)
    for name, r in dataset_raags().items():
        by_scale[frozenset(r.scale)].append(name)
    return {k: sorted(v) for k, v in by_scale.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["m3", "m10"])
    ap.add_argument("--tonics", nargs="+", default=["video", "true"])
    args = ap.parse_args()

    groups = twin_groups()
    twin_of = {}
    for members in groups.values():
        for m in members:
            twin_of[m] = set(members) - {m}
    print(f"{len(groups)} scale-sharing groups covering {len(twin_of)} raags:")
    for members in sorted(groups.values(), key=lambda v: -len(v)):
        print("   ", ", ".join(members))
    print()

    from tonic_ablation import METHODS
    hdr = f"{'method':8s} {'tonic':8s} {'twin-clip top1':>14s} {'-> a twin':>10s} {'all-clip top1':>14s}"
    print(hdr); print("-" * len(hdr))
    for name in args.methods:
        method_kw, extra = METHODS[name]
        for tonic in args.tonics:
            rep = Params(**BASE_REP, tonic_mode=tonic)
            feats = train_feats(rep, {}, extra)
            fold = group_folds([f.clip for f in feats])
            m = make_method(name, **method_kw)
            hit = tot = twin_hit = twin_tot = twin_to_twin = 0
            for f in feats:
                if m.fitted:  # not used by m3/m10, kept so the harness stays honest
                    continue
                pred = m.predict(f)
                true = f.clip.raag
                hit += pred == true; tot += 1
                if true in twin_of:
                    twin_tot += 1
                    twin_hit += pred == true
                    twin_to_twin += pred in twin_of[true]
            print(f"{name:8s} {tonic:8s} {twin_hit/max(twin_tot,1):14.3f} "
                  f"{twin_to_twin/max(twin_tot,1):10.3f} {hit/max(tot,1):14.3f}"
                  f"   (n_twin={twin_tot})")


if __name__ == "__main__":
    main()
