"""S7a: fit the reader on the notation corpus, and measure it with recordings held out.

    poetry run python fit_reader.py            # cross-validated comparison
    poetry run python fit_reader.py --save     # also fit on all notation -> results/reader.json

Learned from notation only:

  swar_offsets   where each swar actually sits, from the notes Neeraja heard (komal swars sit sharp
                 of equal temperament). Shrunk toward 0 when a swar is rare.
  onset_cost     what it costs to declare a new note, separately for slow and fast stretches --
                 alap over-reads, taan under-reads, and one constant cannot serve both.
  constants      (2026-09-30) every other constant of the reader -- pitch tolerance, ornament
                 prices, minimum note length, what counts as held -- by coordinate ascent on
                 misread rate over CONST_GRID.

`--save` keeps whichever variant has the lowest *held-out* misread rate, so adding the constants
fit can only be adopted if it generalises across recordings. Consumers call `load()`.

Tempo is measured from the contour alone (held notes per second), so the fitted reader can be
applied to audio nobody has notated.

Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import corpus
import decode
import matcher
import notes
from metrics import edit_ops

SHRINK_N = 10                        # an offset from n notes is scaled by n / (n + SHRINK_N)
FAST_NOTES_PER_S = 1.0               # held-note rate above which a stretch reads as fast
ONSET_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
READER_JSON = C.RESULTS_DIR / "reader.json"
CONST_GRID = {                          # values tried per constant; the current one is always kept
    "free_cents": (15.0, 23.0, 35.0, 50.0), "scale_cents": (50.0, 70.0, 100.0),
    "note_cap": (1.5, 2.0, 3.0), "orn_cost": (0.3, 0.6, 1.0, 1.5),
    "transit_cost": (0.05, 0.1, 0.3), "min_dwell_s": (0.03, 0.05, 0.07, 0.10),
    "held_slope": (400.0, 800.0, 1200.0), "kan_cents": (100.0, 200.0, 300.0),
    "onset_slow": ONSET_GRID, "onset_fast": ONSET_GRID,
}
CONST_ROUNDS = 3


def load():
    """(params, (onset_slow, onset_fast)) of the saved reader. `params` holds *every* constant the
    reader uses (since 2026-10-03), so editing config.MATCH no longer changes the reader silently."""
    r = json.loads(READER_JSON.read_text())
    params = r.get("params") or dict(C.READ_MATCH, swar_offsets=r["swar_offsets"])
    return {**C.MATCH, **params}, (r["onset_slow"], r["onset_fast"])


def density(cents, hop, params=None):
    """Held notes per second: the tempo of a stretch, from its contour alone."""
    held = matcher._held(cents, hop, {**C.MATCH, **(params or {})})
    onsets = int(np.sum(held[1:] & ~held[:-1]) + (1 if len(held) and held[0] else 0))
    return onsets / max(len(cents) * hop, 1e-6)


def fit_offsets(stretches, params):
    """Per-swar median deviation of the notated notes from equal temperament."""
    devs = {s: [] for s in range(12)}
    for st in stretches:
        if st["method"] != "align":
            continue                           # hand-spaced notes carry no pitch evidence
        kinds, _ = decode.align(st["cents"], st["swars"], st["hop"], params=params, free_edges=True)
        for k, sw in enumerate(st["swars"]):
            v = st["cents"][kinds == k]
            v = v[~np.isnan(v)]
            if len(v):
                devs[sw % 12].append((np.median(v) - 100.0 * sw + 600.0) % 1200.0 - 600.0)
    return [float(np.median(d) * len(d) / (len(d) + SHRINK_N)) if d else 0.0
            for s, d in sorted(devs.items())]


def read(st, params, onsets=None):
    """The reader's swar sequence for one stretch; `onsets` = (slow, fast) makes it tempo-aware."""
    p = dict(params)
    if onsets is not None:
        p["onset_cost"] = notes.onset_cost(st["cents"], st["hop"], params, onsets)
    seq, _ = decode.free_read(st["cents"], st["hop"], params=p)
    return seq


def misread(stretches, params, onsets=None):
    ops = np.zeros(3, int)
    n = 0
    for st in stretches:
        human = [s % 12 for s in st["swars"]]
        ops += np.array(edit_ops(human, read(st, params, onsets)))
        n += len(human)
    return ops, n


def fit_onsets(stretches, params):
    """Best onset cost for slow and for fast stretches, each by its own misread rate."""
    slow = [s for s in stretches if density(s["cents"], s["hop"], params) < FAST_NOTES_PER_S]
    fast = [s for s in stretches if density(s["cents"], s["hop"], params) >= FAST_NOTES_PER_S]
    best = []
    for group in (slow, fast):
        scores = []
        for o in ONSET_GRID:
            ops, n = misread(group, dict(params, onset_cost=o))
            scores.append(ops.sum() / max(n, 1))
        best.append(ONSET_GRID[int(np.argmin(scores))])
    return tuple(best)


def rate(stretches, params, onsets):
    ops, n = misread(stretches, params, onsets)
    return ops.sum() / max(n, 1)


def fit_constants(train, params, onsets):
    """Coordinate ascent over CONST_GRID (onset costs included), minimising misread on `train`."""
    best_p, best_o = dict(params), tuple(onsets)
    best = rate(train, best_p, best_o)
    for _ in range(CONST_ROUNDS):
        improved = False
        for k, values in CONST_GRID.items():
            for v in values:
                if k == "onset_slow":
                    p, o = best_p, (v, best_o[1])
                elif k == "onset_fast":
                    p, o = best_p, (best_o[0], v)
                else:
                    p, o = dict(best_p, **{k: v}), best_o
                if p == best_p and o == best_o:
                    continue
                r = rate(train, p, o)
                if r < best - 1e-4:
                    best_p, best_o, best, improved = p, o, r, True
        if not improved:
            break
    return best_p, best_o


def variants(train, fit_all=True):
    base = dict(C.READ_MATCH)
    offsets = fit_offsets(train, dict(C.NOTATE_MATCH))
    with_off = dict(base, swar_offsets=offsets)
    both = (with_off, fit_onsets(train, with_off))
    out = {
        "reader as of S6 (onset 2.0, equal temperament)": (base, None),
        "+ per-swar centres": (with_off, None),
        "+ onset by tempo": (base, fit_onsets(train, base)),
        "+ both": both,
    }
    if fit_all:
        out["+ both + all constants"] = fit_constants(train, *both)
    return out, offsets


def cross_validate(k=4):
    st, fold = corpus.folds(k)
    fold = np.array(fold)
    totals = {}
    print(f"{len(st)} stretches, {len({s['recording'] for s in st})} recordings, {k} folds "
          "(no recording on both sides)\n")
    for f in range(k):
        train = [s for s, g in zip(st, fold) if g != f]
        held = [s for s, g in zip(st, fold) if g == f]
        vs, _ = variants(train)
        for name, (params, onsets) in vs.items():
            ops, n = misread(held, params, onsets)
            t = totals.setdefault(name, [np.zeros(3, int), 0, 0, 0])
            t[0] += ops; t[1] += n
            t[2] += sum(len(read(s, params, onsets)) for s in held)
        print(f"  fold {f + 1}/{k} done ({len(held)} held-out stretches)", flush=True)

    print(f"\n{'held out, pooled over folds':46s} {'read/notated':>13s} {'sub':>5s} {'del':>5s} "
          f"{'ins':>5s} {'misread':>8s}")
    for name, (ops, n, n_read, _) in totals.items():
        print(f"  {name:44s} {n_read:6d}/{n:<6d} {ops[0]:5d} {ops[1]:5d} {ops[2]:5d} "
              f"{ops.sum() / n:8.3f}")
    return {name: t[0].sum() / t[1] for name, t in totals.items()}


def save(cv):
    """Refit the variant with the lowest held-out misread rate on all notation."""
    choice = min(cv, key=cv.get)
    st = corpus.stretches()
    vs, offsets = variants(st, fit_all=choice.endswith("all constants"))
    params, onsets = vs[choice]
    onsets = onsets or (params["onset_cost"], params["onset_cost"])
    READER_JSON.parent.mkdir(exist_ok=True)
    READER_JSON.write_text(json.dumps(dict(
        variant=choice, cv_misread=round(cv[choice], 4),
        params={**C.MATCH, **params},                       # every constant, frozen
        swar_offsets=params.get("swar_offsets", offsets), onset_slow=onsets[0], onset_fast=onsets[1],
        fast_notes_per_s=FAST_NOTES_PER_S, fitted_on="notation corpus, all stretches",
        n_stretches=len(st), n_swars=sum(len(s["swars"]) for s in st)), indent=1))
    print(f"\nchosen by held-out misread: {choice} ({cv[choice]:.3f})")
    print("  constants: " + ", ".join(f"{k}={v}" for k, v in params.items()
                                      if k != "swar_offsets" and C.READ_MATCH.get(k, C.MATCH.get(k)) != v))
    from utils import raagdb
    print("\nfitted on all notation:")
    print("  swar centres (cents from equal temperament): "
          + "  ".join(f"{raagdb.SWAR_NAMES[s]} {o:+.0f}" for s, o in enumerate(offsets)))
    print(f"  onset cost: slow {onsets[0]}, fast {onsets[1]}   -> {READER_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    cv = cross_validate()
    if a.save:
        save(cv)
