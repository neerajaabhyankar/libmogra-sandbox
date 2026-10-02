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
                             nearest of the 12 swars. Nothing fitted to notation
  tuned heuristic notes      the reader (decode.free_read) with its onset cost and swar centres
                             fitted to notation (results/reader.json)
Recordings are cut into phrases at silences; direction is never judged across a silence.

**Audio only** (since 2026-10-03): neither method knows the raag -- no scale restriction. The raag
label only says which recordings to pool for a test2 question. Earlier results (test2_s10.json,
and test2.json before this date) restricted notes to the raag's scale.

Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import decode
import fit_reader
import fullaudio
import matcher
from insights import core
from utils import raagdb

OUT = C.RESULTS_DIR / "test2.json"
MIN_HELD_S = 0.10          # held-notes method: a held stretch shorter than this is not a note
MIN_PHRASE_S = 1.0         # phrases shorter than this carry no direction worth counting
HELD = "held-notes only (untuned)"
READER = "tuned heuristic notes"


def phrases(ctr):
    """(a, b) frame bounds of voiced stretches, split at unvoiced gaps longer than max_gap_s."""
    return core.phrases(ctr.cents, ctr.hop, C.MATCH["max_gap_s"], MIN_PHRASE_S)


def _snap(c, scale):
    """Nearest scale swar to `c` cents, as (swar 0-11, absolute cents of that swar)."""
    cand = [(abs(c - (100 * s + 1200 * o)), s, 100 * s + 1200 * o)
            for s in scale for o in (-1, 0, 1, 2)]
    _, s, abs_c = min(cand)
    return s, abs_c


def held_notes(cents, hop, scale):
    held = matcher._held(cents, hop, C.MATCH)
    n = int(MIN_HELD_S / hop)
    out, t = [], 0
    while t < len(held):
        if held[t]:
            u = t
            while u < len(held) and held[u]:
                u += 1
            if u - t >= n:
                out.append(_snap(float(np.nanmedian(cents[t:u])), scale))
            t = u
        else:
            t += 1
    return out


def reader_notes(cents, hop, scale, params, onsets):
    return [(sw, c) for sw, c, _, _ in core.notes(cents, hop, scale, (params, onsets))]


directions = core.directions    # [up, down] per swar, judged by the next note


def measure(raag):
    """Pool every recording of `raag`; each is read without knowing its raag."""
    scale = list(range(12))
    params, onsets = fit_reader.load()
    counts = {HELD: {}, READER: {}}
    videos = fullaudio.cached_videos(tuple([raag]))
    for v in videos:
        ctr = fullaudio.contour(v)
        for a, b in phrases(ctr):
            seg = ctr.cents[a:b]
            for m, notes in ((HELD, held_notes(seg, ctr.hop, scale)),
                             (READER, reader_notes(seg, ctr.hop, None, params, onsets))):
                directions(notes, counts[m])
        print(f"  {raag:12s} {v}", flush=True)
    return counts, len(videos)


def main():
    truth = json.loads(C.UNIDIR_JSON.read_text())["raags"]
    names = raagdb.SWAR_NAMES
    rows = []
    for e in truth:
        counts, n_rec = measure(e["raag"])
        for sw, lab in e["swars"].items():
            s = names.index(sw)
            row = dict(raag=e["raag"], swar=sw, aaroh=lab["aaroh"], avaroh=lab["avaroh"],
                       control=lab["control"], recordings=n_rec)
            for m, c in counts.items():
                up, down = c.get(s, [0, 0])
                row[m] = dict(up=up, down=down, up_frac=up / max(up + down, 1))
            rows.append(row)
            print(f"{e['raag']:12s} {sw:2s} aaroh {'y' if lab['aaroh'] else 'n'} avaroh "
                  f"{'y' if lab['avaroh'] else 'n'}   " + "   ".join(
                      f"{row[m]['up_frac']:.2f}" for m in counts))
    print("up-fraction columns: " + " | ".join(counts))
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
