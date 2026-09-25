"""The data-discipline check. Run it before trusting any number.

    poetry run python audit.py

This module, not prose, decides which human labels are training, validation or test. Anything
that fits parameters must import `splits()` and honour it; `DATA.md` explains the terms and the
reasoning, but this file is what the code actually obeys.

The rules, in one place:

  R1  A *judgment* (a y/n on a candidate span) is TEST when its recording carries no notation.
  R2  It is VALIDATION when the recording is notated but the two never overlap in time, with
      GUARD_S seconds of margin -- notation teaches the model about a moment, not a recording.
  R3  It is UNUSABLE when it overlaps a notated stretch. Nothing fits on these; nothing scores.
  R4  Raags in TEST_ONLY_RAAGS carry no notation at all, so all their judgments are TEST.
  R5  A pool is never rebuilt once it carries judgments: labels are keyed by *index into* the
      pool, so regenerating it silently re-points them. (`pool.py` refuses without --force.)
"""

import json
from collections import Counter, defaultdict

import _bootstrap  # noqa: F401
import config as C

GUARD_S = 5.0       # a judgment this close to a notated stretch is not treated as unseen


def _last(path, key):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r[key]] = r
    return out


def chunks():
    return {c["id"]: c for c in json.loads((C.S3_DIR / "chunks.json").read_text())}


def notations():
    """{chunk_id: record} for chunks that still exist and have stretches."""
    if not C.NOTATIONS.exists():
        return {}
    ch = chunks()
    return {k: v for k, v in _last(C.NOTATIONS, "chunk_id").items()
            if k in ch and v.get("segments")}


def judgments():
    """One record per (samooha, candidate) -- the last verdict given."""
    path = C.S3_DIR / "labels.jsonl"
    if not path.exists():
        return []
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("pool") == C.POOL_VERSION:
            out[(r["phrase_id"], r["index"])] = r
    return list(out.values())


def notated_spans():
    """{recording: [(start, end), ...]} in recording time, from every notated stretch."""
    ch, out = chunks(), defaultdict(list)
    for cid, rec in notations().items():
        c = ch[cid]
        for s in rec["segments"]:
            out[c["video"]].append((c["t0"] + s["t0"], c["t0"] + s["t1"]))
    return out


def splits(guard_s=GUARD_S):
    """Every judgment labelled 'test', 'validation' or 'unusable'. The rules R1-R4 live here."""
    spans = notated_spans()
    out = {"test": [], "validation": [], "unusable": []}
    for j in judgments():
        if j["video"] not in spans:
            out["test"].append(j)                                    # R1, and R4 by construction
        elif any(j["t0"] < b + guard_s and a - guard_s < j["t1"] for a, b in spans[j["video"]]):
            out["unusable"].append(j)                                # R3
        else:
            out["validation"].append(j)                              # R2
    return out


def main():
    ch, notes, js = chunks(), notations(), judgments()
    segs = [s for r in notes.values() for s in r["segments"]]
    swars = sum(len(s["swars"].split()) for s in segs)
    print("TRAINING -- notation (what a musician heard, written as swars)")
    print(f"  {len(notes)} chunks, {len(segs)} stretches, {swars} swars, "
          f"{sum(s['t1'] - s['t0'] for s in segs):.0f} s")
    print(f"  raags: {', '.join(sorted({ch[c]['raag'] for c in notes}))}")
    print(f"  recordings: {len({ch[c]['video'] for c in notes})}")
    by_method = Counter(s.get("method", "align") for s in segs)
    print(f"  {by_method['align']} aligned to the pitch track, {by_method['even']} spaced by hand")

    s = splits()
    print("\nTEST / VALIDATION -- judgments (y/n on one candidate span)")
    print(f"  {len(js)} judged so far over {len({j['phrase_id'] for j in js})} samoohas")
    for name in ("test", "validation", "unusable"):
        rows = s[name]
        if not rows:
            print(f"  {name:11s} {0:4d}")
            continue
        yes = sum(1 for j in rows if j["verdict"] == "yes")
        print(f"  {name:11s} {len(rows):4d}   {len({j['phrase_id'] for j in rows})} samoohas, "
              f"{len({j['video'] for j in rows})} recordings, {yes / len(rows):.0%} yes")

    print("\n  per samooha (test / validation):")
    for pid in sorted({j["phrase_id"] for j in js}):
        t = sum(1 for j in s["test"] if j["phrase_id"] == pid)
        v = sum(1 for j in s["validation"] if j["phrase_id"] == pid)
        u = sum(1 for j in s["unusable"] if j["phrase_id"] == pid)
        print(f"    {pid:22s} {t:3d} / {v:3d}" + (f"   ({u} unusable)" if u else ""))

    import mukhyangas
    pools = {p.stem for p in (C.S3_DIR / "pool").glob("*.json")}
    waiting = [p for p in mukhyangas.load()
               if p.slug in pools and not any(j["phrase_id"] == p.id for j in js)]
    missing = [p for p in mukhyangas.load() if p.slug not in pools]
    if waiting:
        n = sum(len(json.loads((C.S3_DIR / "pool" / f"{p.slug}.json").read_text())["items"])
                for p in waiting)
        print(f"\n  waiting to be judged: {len(waiting)} samoohas, {n} candidates "
              f"({', '.join(p.id for p in waiting)})")
    if missing:
        print(f"  no pool yet: {', '.join(p.id for p in missing)}")

    print("\nCHECKS")
    ok = True
    bad_raags = [c for c in notes if ch[c]["raag"] in getattr(C, "TEST_ONLY_RAAGS", [])]
    ok &= _check("R4  no notation in a test-only raag", not bad_raags, bad_raags)
    leaked = [j for j in s["test"] + s["validation"]
              if any(j["t0"] < b + GUARD_S and a - GUARD_S < j["t1"]
                     for a, b in notated_spans().get(j["video"], []))]
    ok &= _check("R3  nothing scored or fitted overlaps a notated stretch", not leaked,
                 [f"{j['phrase_id']}#{j['index']}" for j in leaked])
    if s["unusable"]:
        print(f"        ({len(s['unusable'])} judgment(s) set aside by this rule, as intended: "
              + ", ".join(f"{j['phrase_id']}#{j['index']}" for j in s["unusable"]) + ")")
    bad_idx = []
    for j in js:
        slug = j["phrase_id"].replace("#", "_")
        f = C.S3_DIR / "pool" / f"{slug}.json"
        if not f.exists():
            bad_idx.append(f"{j['phrase_id']}: pool missing")
        else:
            items = json.loads(f.read_text())["items"]
            if j["index"] >= len(items) or items[j["index"]]["video"] != j["video"]:
                bad_idx.append(f"{j['phrase_id']}#{j['index']}: pool no longer matches the label")
    ok &= _check("R5  every judgment still points at the candidate it judged", not bad_idx, bad_idx)
    print("\n" + ("all good" if ok else "SOMETHING IS WRONG -- see above"))


def _check(label, passed, offenders):
    print(f"  {'ok  ' if passed else 'FAIL'}  {label}")
    for o in list(offenders)[:5]:
        print(f"          {o}")
    return passed


if __name__ == "__main__":
    main()
