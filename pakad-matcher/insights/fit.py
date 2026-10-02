"""Tune the insight functions on the notation corpus (training only).

    poetry run python -m insights.fit      # -> results/insights/fit.json

  nyas swar     at every pause inside a notated stretch (unvoiced >= pause_min_s between two
                different notated notes), the notated swar before it is the answer. Grid over the
                final-note rule (tail_trim_s, skip_short_s, droop_only); scored with recordings held out.
  directions    per (raag, swar), up/down counted from the notation vs from the tuned heuristic notes
                on the same stretches; sweep dir_min_note_s.

Audio only: notes are read with no scale (raag labels only group the notation counts).

Which gaps are *pauses* is not tuned: in the notation, gaps inside one notated note (dropouts) are
not shorter than gaps between notes, so the notation cannot say. Defaults in config.INSIGHTS.

Terms: [DATA.md § Glossary](../DATA.md#glossary).
"""

import itertools
import json

import numpy as np
from scipy.stats import spearmanr

import _bootstrap  # noqa: F401
import config as C
import corpus
import decode
import fit_reader
from insights import core

RULE_GRID = dict(tail_trim_s=(0.0, 0.05, 0.1), skip_short_s=(0.0, 0.05, 0.1, 0.15, 0.2),
                 droop_only=(True, False))
DIR_GRID = (0.0, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3)
LOOKBACK_S = 3.0           # read this much contour before a pause
ONE_DIR = 0.85             # in the notation, >= 85% one way = one-directional
MIN_MOVES = 8              # raag-swars with fewer notated moves are not compared
K = 4


def notated_pauses(st):
    """[(stretch index, gap start frame, notated swar before)] for gaps between different notes."""
    out = []
    for i, s in enumerate(st):
        if s["method"] != "align":
            continue
        c = np.asarray(s["cents"], float)
        kinds, _ = decode.align(c, s["swars"], s["hop"], params=dict(C.NOTATE_MATCH), free_edges=True)
        v, t = ~np.isnan(c), 0
        while t < len(c):
            if v[t]:
                t += 1; continue
            u = t
            while u < len(c) and not v[u]:
                u += 1
            before, after = kinds[:t][kinds[:t] >= 0], kinds[u:][kinds[u:] >= 0]
            if (u - t) * s["hop"] >= C.INSIGHTS["pause_min_s"] and len(before) and len(after) \
                    and before[-1] != after[0] and u < len(c):
                out.append((i, t, s["swars"][before[-1]] % 12))
            t = u
    return out


def fit_nyas(st, reader):
    rows = notated_pauses(st)
    cache = {}

    def notes_before(i, t, trim):
        if (i, t, trim) not in cache:
            s = st[i]
            seg = np.asarray(s["cents"], float)[max(0, t - int(LOOKBACK_S / s["hop"])):t]
            b = len(seg) - int(trim / s["hop"])
            cache[(i, t, trim)] = core.notes(seg[:b], s["hop"], None, reader) \
                if b > 4 else []
        return cache[(i, t, trim)]

    hits = {}
    for vals in itertools.product(*RULE_GRID.values()):
        rule = dict(zip(RULE_GRID, vals))
        hits[vals] = np.array([core.final_swar(notes_before(i, t, rule["tail_trim_s"]), rule) == sw
                               for i, t, sw in rows])
    longest = np.mean([bool(ns) and max(ns, key=lambda n: n[3] - n[2])[0] == sw
                       for (i, t, sw) in rows for ns in [notes_before(i, t, 0.0)]])
    recs = sorted({st[i]["recording"] for i, _, _ in rows})
    fold_of = {r: j % K for j, r in enumerate(np.random.default_rng(0).permutation(recs))}
    fold = np.array([fold_of[st[i]["recording"]] for i, _, _ in rows])
    held = np.zeros(len(rows), bool)
    for k in range(K):
        pick = max(hits, key=lambda v: hits[v][fold != k].mean())
        held[fold == k] = hits[pick][fold == k]
    best = max(hits, key=lambda v: hits[v].mean())
    kinds = {}
    for (i, _, _), h in zip(rows, held):
        kinds.setdefault(st[i]["kind"], []).append(h)
    res = dict(pauses=len(rows), recordings=len(recs), rule=dict(zip(RULE_GRID, best)),
               held_out_accuracy=round(float(held.mean()), 3),
               plain_last_note=round(float(hits[(0.0, 0.0, True)].mean()), 3),
               longest_note=round(float(longest), 3),
               by_kind={k: round(float(np.mean(v)), 3) for k, v in kinds.items()})
    print(f"nyas swar: {len(rows)} notated pauses, {len(recs)} recordings")
    print(f"  longest note before the pause   {res['longest_note']:.3f}")
    print(f"  plain last note                 {res['plain_last_note']:.3f}")
    print(f"  tuned rule, recordings held out {res['held_out_accuracy']:.3f}   {res['by_kind']}")
    print(f"  rule fitted on all: {res['rule']}")
    return res


def fit_directions(st, reader):
    human, read = {}, {}
    for s in st:
        core.directions([(sw % 12, 100 * (sw % 12) + 1200 * o, 0, 0)
                         for sw, o in zip(s["swars"], s["octaves"])], human.setdefault(s["raag"], {}))
        read.setdefault(s["raag"], []).append(
            core.notes(np.asarray(s["cents"], float), s["hop"], None, reader))
    keys = [(r, w) for r in human for w, (u, d) in human[r].items() if u + d >= MIN_MOVES]
    lab = {k: np.sign(human[k[0]][k[1]][0] - human[k[0]][k[1]][1])
           for k in keys if max(human[k[0]][k[1]]) / sum(human[k[0]][k[1]]) >= ONE_DIR}
    print(f"\ndirections: {len(keys)} raag-swars with >= {MIN_MOVES} notated moves, "
          f"{len(lab)} one-directional in the notation")
    rows = []
    for mn in DIR_GRID:
        m = {}
        for r in read:
            m[r] = {}
            for ns in read[r]:
                core.directions([n for n in ns if n[3] - n[2] >= mn], m[r])
        frac = lambda c: c[0] / max(c[0] + c[1], 1)
        rho = spearmanr([frac(human[r][w]) for r, w in keys],
                        [frac(m[r].get(w, [0, 0])) for r, w in keys]).correlation
        got = [m[r].get(w, [0, 0]) for r, w in lab]
        right = np.mean([np.sign(u - d) == lab[k] for k, (u, d) in zip(lab, got)])
        share = np.median([max(u, d) / max(u + d, 1) for u, d in got])
        rows.append(dict(min_note_s=mn, rho=round(rho, 3), direction_right=round(right, 3),
                         majority_share=round(share, 3)))
        print(f"  min note {mn:.2f}s   rho(up-fraction) {rho:.2f}   one-directional: "
              f"direction right {right:.2f}, median majority share {share:.2f}")
    return rows


def main():
    st, reader = corpus.stretches(), fit_reader.load()
    out = dict(nyas=fit_nyas(st, reader), directions=fit_directions(st, reader))
    C.INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.INSIGHTS_DIR / "fit.json").write_text(json.dumps(out, indent=1))
    print(f"-> {C.INSIGHTS_DIR / 'fit.json'}")


if __name__ == "__main__":
    main()
