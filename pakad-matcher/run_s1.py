"""S1 driver: run the matcher for every kept phrase of the focus raags over their train clips.

    poetry run python run_s1.py [--raags Bageshree Shree ...]

Writes results/s1/{candidates.csv, summary.csv, plots/<phrase>.png, audio/<phrase>_<rank>.wav}.
"""

import argparse
import csv

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import contour
import matcher
import phrases
import plot
from utils import raagdb

FIELDS = ["phrase_id", "raag", "phrase", "clip_id", "video", "t0", "t1",
          "cost", "pitch_cost", "orn_frac", "gap_frac"]


def diverse(items, n, cap):
    out, per = [], {}
    for it in items:
        v = contour.clips()[it[0].clip_id].video
        if per.get(v, 0) < cap:
            out.append(it); per[v] = per.get(v, 0) + 1
        if len(out) == n:
            break
    return out


def run_phrase(p, clip_ids):
    items, best = [], []
    for cid in clip_ids:
        ctr = contour.contour(cid)
        cands = matcher.match(ctr, p.swars)
        items += [(ctr, c) for c in cands]
        best.append(cands[0].cost if cands else np.inf)
    items.sort(key=lambda x: x[1].cost)
    return items, np.array(best)


def main(raags):
    for d in ("plots", "audio"):
        (C.S1_DIR / d).mkdir(parents=True, exist_ok=True)
    cached = set(contour.cached_ids())
    rows, summary = [], []
    for p in phrases.kept_phrases(raags):
        ids = [c.clip_id for c in contour.clips().values() if c.raag == p.raag and c.clip_id in cached]
        items, best = run_phrase(p, ids)
        for ctr, c in items:
            rows.append([p.id, p.raag, p.text, ctr.clip_id, contour.clips()[ctr.clip_id].video,
                         round(c.t0, 2), round(c.t1, 2), round(c.cost, 3), round(c.pitch_cost, 3),
                         round(c.orn_frac, 3), round(c.gap_frac, 3)])
        top = diverse(items, C.S1_PLOT_TOP, C.S1_MAX_PER_VIDEO)
        mid = items[len(items) // 2: len(items) // 2 + C.S1_PLOT_MID]
        scale = raagdb.dataset_raags([p.raag])[p.raag].scale
        plot.grid(top + mid, p, scale, C.S1_DIR / "plots" / f"{p.id.replace('#', '_')}.png")
        for i, (ctr, c) in enumerate(top[:C.S1_AUDIO_TOP]):
            plot.snippet(contour.clips()[ctr.clip_id], c.t0, c.t1,
                         C.S1_DIR / "audio" / f"{p.id.replace('#', '_')}_{i}.wav")
        lo, hi = C.S1_COST_BANDS
        summary.append([p.id, p.text, p.idf, len(ids), round(float(np.median(best)), 2),
                        int((best < lo).sum()), int((best < hi).sum())])
        print(f"{p.id:20s} {p.text:26s} clips={len(ids):3d} median best={np.median(best):.2f} "
              f"<{lo}: {(best < lo).sum():3d}  <{hi}: {(best < hi).sum():3d}", flush=True)

    with open(C.S1_DIR / "candidates.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([FIELDS] + rows)
    with open(C.S1_DIR / "summary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["phrase_id", "phrase", "idf", "clips", "median_best_cost",
                                   f"clips_best_lt_{lo}", f"clips_best_lt_{hi}"]] + summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raags", nargs="+", default=C.FOCUS_RAAGS)
    main(ap.parse_args().raags)
