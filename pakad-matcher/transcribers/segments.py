"""The time ranges a pitch source must cover, per recording, for each purpose.

    notation     every notated chunk (the reader is fitted here; direction proxy labels too)
    validation   every validation judgment's span (phrase method choice)
    insights     training and validation insight clips (insight method choice)
    test         test judgments' spans and test insight clips -- ONLY for the final pick, once

Ranges are padded by config.SEGMENT_PAD_S and merged. Wrong-tonic recordings are left out (R7).
No contour is read here, so this never depends on the source being prepared.
"""

import audit
import config as C
from insights import clips

PURPOSES = ("notation", "validation", "insights")


def _raw(purpose):
    ch, sp = audit.chunks(), audit.splits()
    if purpose == "notation":
        return [(ch[c]["video"], ch[c]["t0"], ch[c]["t1"]) for c in audit.notations()]
    if purpose in ("validation", "test"):
        out = [(j["video"], j["t0"], j["t1"]) for j in sp[purpose]]
        if purpose == "test":
            out += [(c["video"], c["t0"], c["t1"]) for c in clips.registry() if c["split"] == "test"]
        return out
    if purpose == "insights":
        return [(c["video"], c["t0"], c["t1"]) for c in clips.registry()
                if c["split"] in ("train", "validation")]
    raise ValueError(purpose)


def ranges(purposes=PURPOSES):
    """{video: [(t0, t1), ...]} merged, padded, sorted."""
    by = {}
    for p in purposes:
        for v, t0, t1 in _raw(p):
            if v not in C.BAD_TONIC_VIDEOS:
                by.setdefault(v, []).append((max(0.0, t0 - C.SEGMENT_PAD_S), t1 + C.SEGMENT_PAD_S))
    out = {}
    for v, rs in by.items():
        merged = []
        for a, b in sorted(rs):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        out[v] = [(round(a, 2), round(b, 2)) for a, b in merged]
    return out


if __name__ == "__main__":
    import sys
    rs = ranges(tuple(sys.argv[1:]) or PURPOSES)
    secs = sum(b - a for r in rs.values() for a, b in r)
    print(f"{len(rs)} recordings, {sum(map(len, rs.values()))} ranges, {secs / 60:.1f} min of audio")
