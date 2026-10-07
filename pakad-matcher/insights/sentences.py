"""I7: count directions within "sentences" (between nyas) instead of within Melodia breaths.

    poetry run python -m insights.sentences

Today a breath is any Melodia gap > 0.25 s (notes.breath_spans), and the direction rule never
counts a move across one. Neeraja (2026-10-06): a breath in the musical sense is the pause after a
nyas, so cut the clip at nyas instead -- stray pitch (an instrument, noise) then no longer makes or
hides a breath, once the nyas detector is good. Here the reader's notes are cut into sentences at:

  Melodia breaths     today's rule (the baseline)
  detected nyas       the frozen nyas choice (learned), refitted per fold
  marked nyas         Neeraja's own nyas windows: an ideal detector. ORACLE -- it reads the
                      held-out clip's labels, so it is a ceiling, never selectable

and the direction threshold rule (core.unidirectional) is retuned per fold for each. Leave-one-
clip-out over train + validation, as insights.evaluate; test is not touched. Terms: DATA.md.
"""

import json

import config as C
from insights import detect, evaluate

SNAP_S = 0.15       # a nyas event belongs to the note that ends within this of it


def sentences(read_, times):
    """The reading re-cut: all notes in order, split after the note ending nearest each time."""
    ns = [n for _, _, x in read_ for n in x]
    cut = set()
    for t in times:
        near = [i for i, n in enumerate(ns) if abs(n[3] - t) <= SNAP_S]
        if near:
            cut.add(min(near, key=lambda i: abs(ns[i][3] - t)))
    out, cur = [], []
    for i, n in enumerate(ns):
        cur.append(n)
        if i in cut:
            out.append((None, None, cur)); cur = []
    return out + ([(None, None, cur)] if cur else [])


def marked_times(it):
    cands = detect.nyas_candidates(it["cents"], it["hop"], it["read"], it["loud"], it["voice"])
    y = detect.label_candidates(cands, it["lab"]["nyas"])
    return [c[1] for c, k in zip(cands, y) if k]


def tune_dir(items):
    best, top = None, -1
    for r in evaluate.DIR_GRID["dir_ratio"]:
        for m in evaluate.DIR_GRID["dir_min_count"]:
            p = dict(C.INSIGHTS, dir_ratio=r, dir_min_count=m)
            s = evaluate.dir_score(items, detect.rule_direction(p))["balanced"]
            if s > top + 1e-9:
                best, top = p, s
    return best


def main():
    items = evaluate.load("train") + evaluate.load("validation")
    print(f"{len(items)} clips (train + validation), leave-one-clip-out\n")
    recut = {"Melodia breaths": {id(it): it["read"] for it in items},
             "detected nyas": {}, "marked nyas (oracle)": {id(it): sentences(it["read"], marked_times(it))
                                                          for it in items}}
    for i, it in enumerate(items):                       # nyas detector never sees its own clip
        m = detect.fit(items[:i] + items[i + 1:])
        recut["detected nyas"][id(it)] = sentences(it["read"], [t for _, t in detect.learned_nyas(m)(it)])
    res = {}
    for name, rd in recut.items():
        its = [dict(it, read=rd[id(it)]) for it in items]
        pred = {}
        for i, it in enumerate(its):
            pred[id(it)] = detect.rule_direction(tune_dir(its[:i] + its[i + 1:]))(it)
        d = evaluate.dir_score(its, lambda it: pred[id(it)])
        n_sent = sum(len(rd[id(it)]) for it in items)
        res[name] = dict(balanced=d["balanced"], recall=d["recall"], sentences=n_sent)
        print(f"  {name:22s} directions {d['balanced']:.3f}  (recall aar {d['recall'].get('aarohi', 0):.2f} "
              f"ava {d['recall'].get('avarohi', 0):.2f} both {d['recall'].get('both', 0):.2f}; "
              f"{n_sent} sentences)")
    frozen = json.loads(detect.CHOICE_JSON.read_text())["cv"]
    print(f"\nfrozen choice for reference (same protocol): " + ", ".join(
        f"{k} {v['directions']:.3f}" for k, v in frozen.items()))
    (C.INSIGHTS_DIR / "sentences_val.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
