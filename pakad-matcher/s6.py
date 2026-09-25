"""S6: score the automatic reading against what Neeraja heard.

    poetry run python s6.py

Two questions, in order of what the project actually needs:

  1. how close is the machine's reading of a stretch to the notation?  (note error rate)
  2. do *aggregates* over the reading agree with aggregates over the notation, even where the
     individual notes do not?  -- this is the one that decides whether statistical questions are
     answerable, and it is the reason 8-in-10 can be enough.
"""

import json
from collections import Counter

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import decode
import fullaudio
from utils import raagdb

SW = raagdb.SWAR_NAMES


def notations(chunk_ids=None):
    """Last save per chunk, for chunks that still exist (some were replaced) and are not empty."""
    last = {}
    for line in open(C.NOTATIONS):
        r = json.loads(line)
        last[r["chunk_id"]] = r
    return {cid: r for cid, r in last.items()
            if r.get("segments") and (chunk_ids is None or cid in chunk_ids)}


def human_swars(text):
    swars, _oct = raagdb.parse_phrase(text.split())
    return [s % 12 for s in swars]


def edit_ops(a, b):
    """Levenshtein between swar sequences; returns (substitutions, deletions, insertions)."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, sub, dele, ins = n, m, 0, 0, 0
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            sub += a[i - 1] != b[j - 1]; i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            dele += 1; i -= 1                      # in the notation, missing from the reading
        else:
            ins += 1; j -= 1                       # in the reading, not in the notation
    return sub, dele, ins


def read_stretch(ch, s, params=None):
    params = params or C.READ_MATCH
    ctr = fullaudio.contour(ch["video"])
    a = int(round((ch["t0"] + s["t0"]) / ctr.hop))
    b = int(round((ch["t0"] + s["t1"]) / ctr.hop))
    cents = ctr.cents[a:b]
    if len(cents) < 4 or np.isnan(cents).all():
        return None
    seq, _spans = decode.free_read(cents, ctr.hop, params=params)
    return seq


def collapse(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def sweep(rows_fn):
    """How much should declaring a new note cost? The corpus can answer that."""
    print("onset  min_dwell   read/notated   sub   del   ins   misread")
    best = None
    for dwell in (0.05, 0.10, 0.15):
        for onset in (0.0, 0.5, 1.0, 2.0, 4.0):
            rows = rows_fn(dict(C.READ_MATCH, onset_cost=onset, min_dwell_s=dwell))
            tot = np.array([edit_ops(r["human"], r["machine"]) for r in rows]).sum(axis=0)
            n_h = sum(len(r["human"]) for r in rows)
            n_m = sum(len(r["machine"]) for r in rows)
            ner = tot.sum() / n_h
            print(f"{onset:5.1f}  {dwell:9.2f}   {n_m:5d}/{n_h:<5d}  {tot[0]:5d} {tot[1]:5d} "
                  f"{tot[2]:5d} {ner:9.2f}{'   <-- best' if best is None or ner < best[0] else ''}")
            if best is None or ner < best[0]:
                best = (ner, onset, dwell)
    print(f"\nbest: onset_cost={best[1]}, min_dwell_s={best[2]}  ->  misread rate {best[0]:.2f}")
    return best


def main():
    chunks = {c["id"]: c for c in json.loads((C.S3_DIR / "chunks.json").read_text())}
    notes = notations(set(chunks))
    rows = []
    for cid, rec in sorted(notes.items()):
        ch = chunks[cid]
        for s in rec["segments"]:
            human = human_swars(s["swars"])
            machine = read_stretch(ch, s)
            if machine is None or not human:
                continue
            rows.append(dict(chunk=cid, kind=ch["kind"], raag=ch["raag"],
                             method=s.get("method", "align"), dur=s["t1"] - s["t0"],
                             human=human, machine=machine))
    print(f"{len(rows)} stretches, {sum(len(r['human']) for r in rows)} notated swars, "
          f"{sum(r['dur'] for r in rows):.0f} s\n")

    print("1. reading a stretch: misread rate = (sub + del + ins) / notated swars")
    print(f"   {'':16s} {'stretches':>9s} {'notated':>8s} {'read':>6s} {'sub':>5s} {'del':>5s} {'ins':>5s} {'misread':>8s}")
    for label, keep in [("everything", lambda r: True),
                        ("alap", lambda r: r["kind"] == "alap"),
                        ("taan", lambda r: r["kind"] == "taan"),
                        ("aligned to track", lambda r: r["method"] == "align"),
                        ("spaced by hand", lambda r: r["method"] == "even")]:
        rs = [r for r in rows if keep(r)]
        if not rs:
            continue
        tot = np.array([edit_ops(r["human"], r["machine"]) for r in rs]).sum(axis=0)
        n_h = sum(len(r["human"]) for r in rs)
        n_m = sum(len(r["machine"]) for r in rs)
        print(f"   {label:16s} {len(rs):9d} {n_h:8d} {n_m:6d} {tot[0]:5d} {tot[1]:5d} {tot[2]:5d} "
              f"{tot.sum() / n_h:8.2f}")

    print("\n   the same, after collapsing repeats on both sides (identity, not rhythm)")
    for label, keep in [("everything", lambda r: True), ("alap", lambda r: r["kind"] == "alap"),
                        ("taan", lambda r: r["kind"] == "taan")]:
        rs = [r for r in rows if keep(r)]
        tot = np.array([edit_ops(collapse(r["human"]), collapse(r["machine"])) for r in rs]).sum(axis=0)
        n_h = sum(len(collapse(r["human"])) for r in rs)
        print(f"   {label:16s} {len(rs):9d} {n_h:8d} {'':6s} {tot[0]:5d} {tot[1]:5d} {tot[2]:5d} "
              f"{tot.sum() / n_h:8.2f}")

    print("\n2. aggregates: does the reading answer the same question as the notation?")
    # (a) which swars are used at all, per chunk
    tp = fp = fn = 0
    for cid in sorted({r["chunk"] for r in rows}):
        rs = [r for r in rows if r["chunk"] == cid]
        H = {x for r in rs for x in r["human"]}
        M = {x for r in rs for x in r["machine"]}
        tp += len(H & M); fp += len(M - H); fn += len(H - M)
    print(f"   swars present in a chunk: recall {tp / (tp + fn):.2f}, precision {tp / (tp + fp):.2f} "
          f"({fn} missed, {fp} spurious over {len(set(r['chunk'] for r in rows))} chunks)")

    # (b) the shape of the swar distribution
    tvs = []
    for cid in sorted({r["chunk"] for r in rows}):
        rs = [r for r in rows if r["chunk"] == cid]
        h = np.array([sum(r["human"].count(s) for r in rs) for s in range(12)], float)
        m = np.array([sum(r["machine"].count(s) for r in rs) for s in range(12)], float)
        if h.sum() and m.sum():
            tvs.append(0.5 * np.abs(h / h.sum() - m / m.sum()).sum())
    print(f"   swar histogram, total-variation distance per chunk: median {np.median(tvs):.2f} "
          f"(0 = identical, 1 = disjoint); {sum(t < 0.25 for t in tvs)}/{len(tvs)} chunks below 0.25")

    # (c) the ascent/descent question, asked of both
    def context(seqs):
        up, down = Counter(), Counter()
        for seq in seqs:
            for prev, cur in zip(seq, seq[1:]):
                step = (cur - prev + 6) % 12 - 6
                (up if step > 0 else down)[cur] += 1
        return up, down
    hu, hd = context([r["human"] for r in rows])
    mu, md = context([r["machine"] for r in rows])
    print("\n   \"is this swar approached from below or above?\" -- notation vs reading")
    print(f"   {'swar':>5s} {'n (notated)':>12s} {'from below':>11s} {'n (read)':>9s} {'from below':>11s}")
    for s in sorted(set(hu) | set(hd), key=lambda s: -(hu[s] + hd[s]))[:8]:
        h_n, m_n = hu[s] + hd[s], mu[s] + md[s]
        if h_n < 10 or m_n < 10:
            continue
        print(f"   {SW[s]:>5s} {h_n:12d} {hu[s] / h_n:11.2f} {m_n:9d} {mu[s] / m_n:11.2f}")


if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        chunks = {c["id"]: c for c in json.loads((C.S3_DIR / "chunks.json").read_text())}
        notes = notations(set(chunks))
        def build(params):
            out = []
            for cid, rec in sorted(notes.items()):
                ch = chunks[cid]
                for s in rec["segments"]:
                    human = human_swars(s["swars"])
                    machine = read_stretch(ch, s, params)
                    if machine is not None and human:
                        out.append(dict(human=human, machine=machine, kind=ch["kind"]))
            return out
        sweep(build)
    else:
        main()
