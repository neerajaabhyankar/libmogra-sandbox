"""The notation corpus as training data, and the judgments as validation/test, in one place.

Everything S7 fits comes through `stretches()`; everything it scores comes through `spans()`.
Splits come from `audit.splits()` -- this module never decides what is test.

    stretches()      notated stretches: contour, the swars heard, the recording they came from
    note_table()     every notated note with its placement and whether the pitch track agrees
                     (python corpus.py --notes -> annotations/notation_notes.jsonl)
    adjusted()       notations.jsonl as stored, each stretch with its aligned notes added
                     (written by --notes too -> annotations/adjusted_notations.jsonl)
    folds(k)         recording-grouped folds over those stretches, for cross-validation
    spans(split)     judged candidate spans for 'validation' or 'test', with their contour
    (scores and intervals: metrics.py)

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


NOTE_TABLE = C.S3_DIR / "notation_notes.jsonl"
F0_AGREE_CENTS = 50.0       # a note's pitch track "agrees" if its median is within this of the swar
F0_MIN_VOICED = 0.5         # ... and at least this share of its frames have a pitch at all


def note_table():
    """One row per notated note: what Neeraja heard, where the aligner placed it (recording
    seconds), and whether the pitch track supports it. Notes the pitch track misses (tanpura Sa
    on top, a tapering voice) are kept -- `f0_agrees` = False marks them, so pitch-based fits can
    skip them and spectrogram methods can still use them. Placement of such a note is the
    aligner's best guess between its neighbours, not a measured boundary."""
    import decode
    ch, rows = audit.chunks(), []
    for cid, rec in sorted(audit.notations().items()):
        c = ch[cid]
        ctr = fullaudio.contour(c["video"])
        for si, seg in enumerate(rec["segments"]):
            toks = seg["swars"].split()
            swars, octs = raagdb.parse_phrase(toks)
            if not swars or len(swars) != len(toks):
                continue
            a = int(round((c["t0"] + seg["t0"]) / ctr.hop))
            b = int(round((c["t0"] + seg["t1"]) / ctr.hop))
            cents = ctr.cents[a:b]
            if len(cents) < 4:
                continue
            if seg.get("method", "align") == "align":
                kinds, _ = decode.align(cents, swars, ctr.hop, params=dict(C.NOTATE_MATCH),
                                        free_edges=True)
            else:
                kinds = np.minimum(np.arange(len(cents)) * len(swars) // len(cents), len(swars) - 1)
            for k, (tok, sw, o) in enumerate(zip(toks, swars, octs)):
                f = np.flatnonzero(kinds == k)
                row = dict(chunk=cid, recording=c["video"], raag=c["raag"], kind=c["kind"],
                           segment=si, index=k, swar=tok, method=seg.get("method", "align"),
                           bad_tonic=c["video"] in C.BAD_TONIC_VIDEOS, t0=None, t1=None,
                           f0_cents=None, voiced=0.0, f0_agrees=False)
                if len(f):
                    v = cents[f][~np.isnan(cents[f])]
                    row.update(t0=round((a + f[0]) * ctr.hop, 3), t1=round((a + f[-1] + 1) * ctr.hop, 3),
                               voiced=round(len(v) / len(f), 2))
                    if len(v):
                        med = float(np.median(v))
                        row["f0_cents"] = round(med, 1)
                        row["f0_agrees"] = bool(row["voiced"] >= F0_MIN_VOICED and abs(
                            (med - 100 * sw + 600) % 1200 - 600) <= F0_AGREE_CENTS)   # pitch class
                rows.append(row)
    return rows


ADJUSTED = C.S3_DIR / "adjusted_notations.jsonl"


def adjusted(rows=None):
    """The notation records (last save per chunk) with `notes` added to each stretch: the
    note_table() placement, in chunk seconds like the stretch's own t0/t1. `notes` is None for a
    stretch note_table() skips (unparsable swars, or under 4 frames)."""
    ch, by = audit.chunks(), {}
    for r in rows or note_table():
        by.setdefault((r["chunk"], r["segment"]), []).append(r)
    out = []
    for cid, rec in sorted(audit.notations().items()):
        c0 = ch[cid]["t0"]
        segs = []
        for si, seg in enumerate(rec["segments"]):
            ns = by.get((cid, si))
            segs.append(dict(seg, notes=None if ns is None else [
                dict(swar=r["swar"], t0=None if r["t0"] is None else round(r["t0"] - c0, 3),
                     t1=None if r["t1"] is None else round(r["t1"] - c0, 3),
                     f0_cents=r["f0_cents"], voiced=r["voiced"], f0_agrees=r["f0_agrees"]) for r in ns]))
        out.append(dict(rec, segments=segs, bad_tonic=ch[cid]["video"] in C.BAD_TONIC_VIDEOS))
    return out


if __name__ == "__main__":
    import sys
    if "--notes" in sys.argv:
        rows = note_table()
        NOTE_TABLE.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ADJUSTED.write_text("".join(json.dumps(r) + "\n" for r in adjusted(rows)))
        n = sum(not r["f0_agrees"] for r in rows)
        print(f"{len(rows)} notes, {n} ({n / len(rows):.0%}) not supported by the pitch track -> {NOTE_TABLE}")
        sys.exit()
    st = stretches()
    print(f"{len(st)} stretches, {sum(len(s['swars']) for s in st)} swars, "
          f"{len({s['recording'] for s in st})} recordings")
    for split in ("validation", "test"):
        sp = spans(split)
        print(f"{split}: {len(sp)} judged spans over {len({s['pid'] for s in sp})} samoohas")
