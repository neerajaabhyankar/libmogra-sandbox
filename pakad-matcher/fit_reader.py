"""S7a: fit the reader on the notation corpus, and measure it with recordings held out.

    poetry run python fit_reader.py            # cross-validated comparison
    poetry run python fit_reader.py --save     # also fit on all notation -> results/reader.json

Two things are learned, both from notation only:

  swar_offsets   where each swar actually sits, from the notes Neeraja heard (komal swars sit sharp
                 of equal temperament). Shrunk toward 0 when a swar is rare.
  onset_cost     what it costs to declare a new note, separately for slow and fast stretches --
                 alap over-reads, taan under-reads, and one constant cannot serve both.

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
from s6 import edit_ops

SHRINK_N = 10                        # an offset from n notes is scaled by n / (n + SHRINK_N)
FAST_NOTES_PER_S = 1.0               # held-note rate above which a stretch reads as fast
ONSET_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
READER_JSON = C.RESULTS_DIR / "reader.json"


def density(cents, hop):
    """Held notes per second: the tempo of a stretch, from its contour alone."""
    held = matcher._held(cents, hop, C.MATCH)
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
        p["onset_cost"] = onsets[1] if density(st["cents"], st["hop"]) >= FAST_NOTES_PER_S else onsets[0]
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
    slow = [s for s in stretches if density(s["cents"], s["hop"]) < FAST_NOTES_PER_S]
    fast = [s for s in stretches if density(s["cents"], s["hop"]) >= FAST_NOTES_PER_S]
    best = []
    for group in (slow, fast):
        scores = []
        for o in ONSET_GRID:
            ops, n = misread(group, dict(params, onset_cost=o))
            scores.append(ops.sum() / max(n, 1))
        best.append(ONSET_GRID[int(np.argmin(scores))])
    return tuple(best)


def variants(train):
    base = dict(C.READ_MATCH)
    offsets = fit_offsets(train, dict(C.NOTATE_MATCH))
    with_off = dict(base, swar_offsets=offsets)
    return {
        "reader as of S6 (onset 2.0, equal temperament)": (base, None),
        "+ per-swar centres": (with_off, None),
        "+ onset by tempo": (base, fit_onsets(train, base)),
        "+ both": (with_off, fit_onsets(train, with_off)),
    }, offsets


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


def save():
    st = corpus.stretches()
    vs, offsets = variants(st)
    params, onsets = vs["+ both"]
    READER_JSON.parent.mkdir(exist_ok=True)
    READER_JSON.write_text(json.dumps(dict(
        swar_offsets=offsets, onset_slow=onsets[0], onset_fast=onsets[1],
        fast_notes_per_s=FAST_NOTES_PER_S, fitted_on="notation corpus, all stretches",
        n_stretches=len(st), n_swars=sum(len(s["swars"]) for s in st)), indent=1))
    from utils import raagdb
    print("\nfitted on all notation:")
    print("  swar centres (cents from equal temperament): "
          + "  ".join(f"{raagdb.SWAR_NAMES[s]} {o:+.0f}" for s, o in enumerate(offsets)))
    print(f"  onset cost: slow {onsets[0]}, fast {onsets[1]}   -> {READER_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    cross_validate()
    if a.save:
        save()
