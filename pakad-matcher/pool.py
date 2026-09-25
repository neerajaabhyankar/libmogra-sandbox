"""Build annotation pools from the full recordings: the matcher's best candidates, in context.

No arbitrary duration cap -- a slow alap rendering of a 3-swar phrase is exactly what we want
to catch. Candidates are simply the matcher's top ones, with at most POOL_TOP_PER_VIDEO from
any recording. Each gets a context window cut at the surrounding silences (the "sentence").

    poetry run python pool.py [--phrase Bageshree#0]
"""

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import fullaudio
import matcher
import mukhyangas
from utils import raagdb


def context_window(ctr, f0, f1):
    """Frame bounds of the musical sentence around [f0, f1], cut at silences."""
    voiced = ~np.isnan(ctr.cents)
    gap = max(1, int(C.CTX_GAP_S / ctr.hop))
    lo_lim, hi_lim = int(C.CTX_MAX_S / ctr.hop), int(C.CTX_MAX_S / ctr.hop)
    min_pad = int(C.CTX_MIN_S / ctr.hop)

    a = max(0, f0 - lo_lim)
    for t in range(f0 - min_pad, a, -1):           # nearest silence before the candidate
        if t >= gap and not voiced[t - gap:t].any():
            a = t
            break
    b = min(len(voiced) - 1, f1 + hi_lim)
    for t in range(f1 + min_pad, b):
        if not voiced[t:t + gap].any():
            b = t
            break
    return a, b


def _spread_over_tempo(shortlist):
    """Interleave the best-by-cost shortlist across tempo, so slow alap renderings are offered
    alongside fast ones. Ranking is still by cost; only the *order of offering* is spread --
    the cheapest candidates are often quick transits, which is a property worth sampling around
    rather than a rule to hard-code."""
    if not shortlist:
        return shortlist
    durs = np.array([c.t1 - c.t0 for _, _, c, _ in shortlist])
    edges = np.quantile(durs, np.linspace(0, 1, C.POOL_TEMPO_BUCKETS + 1)[1:-1])
    buckets = [[] for _ in range(C.POOL_TEMPO_BUCKETS)]
    for item, d in zip(shortlist, durs):
        buckets[int(np.searchsorted(edges, d))].append(item)
    out = []
    while any(buckets):
        for b in buckets:
            if b:
                out.append(b.pop(0))
    return out


def build_phrase(p, videos):
    cands = []
    for v in videos:
        ctr = fullaudio.contour(v)
        for c in matcher.match(ctr, p.swars, top_k=C.POOL_PER_VIDEO_SEARCH, octaves=p.octaves):
            cands.append((c.cost, v, c, ctr))
    cands.sort(key=lambda t: t[0])
    cands = _spread_over_tempo(cands[:C.POOL_SHORTLIST])

    items, per_video = [], {}
    for cost, v, c, ctr in cands:
        if per_video.get(v, 0) >= C.POOL_TOP_PER_VIDEO:
            continue
        per_video[v] = per_video.get(v, 0) + 1
        a, b = context_window(ctr, c.f0, c.f1)
        cents = ctr.cents[a:b + 1]
        items.append(dict(
            video=v, raag=p.raag, rank=len(items),
            t0=round(c.t0, 2), t1=round(c.t1, 2), dur=round(c.t1 - c.t0, 2),
            win_t0=round(a * ctr.hop, 2), win_t1=round((b + 1) * ctr.hop, 2),
            cost=round(c.cost, 3), pitch_cost=round(c.pitch_cost, 3),
            orn_frac=round(c.orn_frac, 3), leaps=c.leaps,
            hop=ctr.hop,
            cents=[None if np.isnan(x) else round(float(x), 1) for x in cents],
            path=[int(x) for x in np.pad(c.path, (c.f0 - a, b - c.f1), constant_values=-2)],
        ))
        if len(items) == C.POOL_PER_PHRASE:
            break
    return items


def build(only=None, force=False):
    """Build pools for phrases that do not have one yet.

    An existing pool is **never** rebuilt without `force`: the judgments in
    `annotations/labels.jsonl` are keyed by (phrase, index into this pool), so regenerating one
    silently re-points every label it carries. Those judgments are the frozen test set.
    """
    (C.S3_DIR / "pool").mkdir(parents=True, exist_ok=True)
    (C.S3_DIR / "audio").mkdir(parents=True, exist_ok=True)
    for p in mukhyangas.load():
        if only and p.id != only:
            continue
        if (C.S3_DIR / "pool" / f"{p.slug}.json").exists() and not force:
            print(f"{p.id:20s} pool exists, leaving it alone (judgments are keyed to it)")
            continue
        videos = [v for v in fullaudio.cached_videos(tuple([p.raag]))]
        if not videos:
            print(f"{p.id:20s} no pitch-tracked recordings for {p.raag} yet"); continue
        items = build_phrase(p, videos)
        scale = sorted(raagdb.dataset_raags([p.raag])[p.raag].scale)
        (C.S3_DIR / "pool" / f"{p.slug}.json").write_text(json.dumps(dict(
            phrase_id=p.id, raag=p.raag, phrase=p.text, source=p.source, note=p.note,
            swars=list(p.swars), scale=scale, matcher=C.MATCHER_VERSION,
            pool=C.POOL_VERSION, items=items)))
        for i, it in enumerate(items):
            _snippet(it, C.S3_DIR / "audio" / f"{p.slug}_{i:02d}.wav")
        print(f"{p.id:20s} {p.text:14s} {len(items):2d} candidates from {len(videos)} recordings, "
              f"cost {min(i['cost'] for i in items):.2f}–{max(i['cost'] for i in items):.2f}, "
              f"match {min(i['dur'] for i in items):.1f}–{max(i['dur'] for i in items):.1f}s", flush=True)


def _snippet(item, path):
    import librosa
    import soundfile as sf
    fa = fullaudio.index()[item["video"]]
    y, sr = librosa.load(fa.path, sr=None, mono=True, offset=item["win_t0"],
                         duration=item["win_t1"] - item["win_t0"])
    sf.write(path, y, sr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild an existing pool (re-points its labels!)")
    a = ap.parse_args()
    build(a.phrase, a.force)
