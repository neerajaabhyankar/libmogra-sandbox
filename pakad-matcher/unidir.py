"""Test2: is a swar used in aaroh, in avaroh, or both? Measured from whole recordings, no labels.

    poetry run python unidir.py            # -> results/test2.json

Ground truth is `neeraja_unidirectionals.json`: per swar, two y/n questions (used in aaroh? used
in avaroh?). A method answers both with one number per swar: the fraction of its occurrences that
go up (aaroh) -- and 1 minus it for avaroh. Pitch is absolute (each note's median), so octave
jumps count the right way.

What "goes up" means is Neeraja's definition of aarohi/avarohi (2026-09-27): **what comes after
X decides it; what comes before does not matter.** X is avarohi if only lower notes follow it,
aarohi if only higher ones do -- e.g. in Vrindavani Sarang, `m P n P N S R n P` is valid.
So each occurrence counts as up or down by its *departure*: the next note.

Two methods, neither sees a test2 label:
  held-notes only (untuned)  every held stretch of the contour (matcher._held), snapped to the
                             nearest of the 12 swars. Not fitted to notation (its held threshold is
                             S4b's, fitted on validation judgments)
  tuned heuristic notes      the reader fitted to notation (notes.notes, results/reader.json)
"Next note" is notes.directions: the next sung note (kan skipped), never across a breath.
Pooled recordings exclude notated ones (training) and wrong-tonic ones (audit R7).

**Audio only** (since 2026-10-03): neither method knows the raag -- no scale restriction. The raag
label only says which recordings to pool for a test2 question. Earlier results (test2_s10.json,
and test2.json before this date) restricted notes to the raag's scale.

Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import json

import numpy as np

import _bootstrap  # noqa: F401
import audit
import config as C
import fit_reader
import fullaudio
import matcher
import metrics
import notes
from utils import raagdb

OUT = C.RESULTS_DIR / "test2.json"
HELD = "held-notes only (untuned)"
READER = "tuned heuristic notes"


def held_notes(cents, hop):
    """Every held stretch (matcher._held, >= held_min_s), snapped to the nearest of the 12 swars:
    [(swar, absolute cents of that swar)]."""
    held = matcher._held(cents, hop, C.MATCH)
    out, t = [], 0
    while t < len(held):
        if held[t]:
            u = t
            while u < len(held) and held[u]:
                u += 1
            c = float(np.nanmedian(cents[t:u]))
            k = int(round(c / 100))
            out.append((k % 12, 100 * k))
            t = u
        else:
            t += 1
    return out


def excluded_recordings():
    """Recordings test2 must not pool: notated ones (training, rule "never straddles") and
    wrong-tonic ones (R7)."""
    ch = audit.chunks()
    return {ch[c]["video"] for c in audit.notations()} | set(C.BAD_TONIC_VIDEOS)


def measure(raag, skip):
    """Pool every usable recording of `raag`; each is read without knowing its raag."""
    reader = fit_reader.load()
    counts = {HELD: {}, READER: {}}
    videos = [v for v in fullaudio.cached_videos(tuple([raag])) if v not in skip]
    for v in videos:
        ctr = fullaudio.contour(v)
        for a, b in notes.breath_spans(ctr.cents, ctr.hop):
            seg = ctr.cents[a:b]
            notes.directions(held_notes(seg, ctr.hop), counts[HELD])
            notes.directions(notes.notes(seg, ctr.hop, reader), counts[READER])
        print(f"  {raag:12s} {v}", flush=True)
    return counts, len(videos)


def auc(rows, m):
    """Pooled AUC over the questions "used in aaroh?" (score: up-fraction) and "used in avaroh?"
    (score: 1 - up-fraction). Two questions per swar, so the swar is the unit of resampling."""
    s = [r[m]["up_frac"] for r in rows] + [1 - r[m]["up_frac"] for r in rows]
    y = np.array([int(r["aaroh"]) for r in rows] + [int(r["avaroh"]) for r in rows])
    s = np.array(s)
    if y.all() or not y.any():
        return np.nan
    a, b = s[y == 1], s[y == 0]
    return float(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))


def main():
    truth = json.loads(C.UNIDIR_JSON.read_text())["raags"]
    names, skip = raagdb.SWAR_NAMES, excluded_recordings()
    rows, left_out = [], 0
    for e in truth:
        left_out += sum(v in skip for v in fullaudio.cached_videos(tuple([e["raag"]])))
        counts, n_rec = measure(e["raag"], skip)
        for sw, lab in e["swars"].items():
            s = names.index(sw)
            row = dict(raag=e["raag"], swar=sw, aaroh=lab["aaroh"], avaroh=lab["avaroh"],
                       control=lab["control"], recordings=n_rec)
            for m, c in counts.items():
                up, down = c.get(s, [0, 0])
                row[m] = dict(up=up, down=down, up_frac=up / max(up + down, 1))
            rows.append(row)
    print(f"\n{len(rows)} swars ({2 * len(rows)} questions); {left_out} recording(s) left out "
          "of the pools (notated or wrong-tonic)")
    out = dict(rows=rows, auc={}, splits_manifest=audit.manifest_hash())
    for m in (HELD, READER):
        out["auc"][m] = metrics.bootstrap(lambda g, m=m: auc([r for x in g for r in x], m),
                                          [[r] for r in rows])
        est, lo, hi = out["auc"][m]
        print(f"  {m:28s} AUC {est:.3f}  95% interval [{lo:.3f}, {hi:.3f}] (swars resampled)")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
