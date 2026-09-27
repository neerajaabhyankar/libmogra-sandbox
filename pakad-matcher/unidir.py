"""Test2: is a swar used in aaroh, in avaroh, or both? Measured from whole recordings, no labels.

    poetry run python unidir.py            # -> results/test2.json

Ground truth is `neeraja_unidirectionals.json`: per swar, two y/n questions (used in aaroh? used
in avaroh?). A method answers both with one number per swar -- the fraction of its occurrences
*approached from below* (aaroh) -- and 1 minus it for avaroh. Approach is judged on absolute pitch
(the median of each note), so octave jumps count the right way.

Two methods, both label-free:
  held notes   every held stretch of the contour (matcher._held), snapped to the nearest scale swar
  reader       the notation-fitted reader (results/reader.json), restricted to the raag's scale
Recordings are cut into phrases at silences; direction is never judged across a silence.
"""

import json

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import decode
import fit_reader
import fullaudio
import matcher
from utils import raagdb

OUT = C.RESULTS_DIR / "test2.json"
MIN_HELD_S = 0.10          # held-notes method: a held stretch shorter than this is not a note
MIN_PHRASE_S = 1.0         # phrases shorter than this carry no direction worth counting


def phrases(ctr):
    """(a, b) frame bounds of voiced stretches, split at unvoiced gaps longer than max_gap_s."""
    voiced = ~np.isnan(ctr.cents)
    gap = max(1, int(C.MATCH["max_gap_s"] / ctr.hop))
    out, a, silent = [], None, 0
    for t, v in enumerate(voiced):
        if v:
            a = t if a is None else a
            silent = 0
        elif a is not None:
            silent += 1
            if silent > gap:
                out.append((a, t - silent + 1)); a, silent = None, 0
    if a is not None:
        out.append((a, len(voiced)))
    return [(a, b) for a, b in out if (b - a) * ctr.hop >= MIN_PHRASE_S]


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
    p = dict(params, onset_cost=onsets[1] if fit_reader.density(cents, hop)
             >= fit_reader.FAST_NOTES_PER_S else onsets[0])
    seq, spans = decode.free_read(cents, hop, params=p, allowed=scale)
    out = []
    for s, (t0, t1) in zip(seq, spans):
        v = cents[int(t0 / hop):int(t1 / hop)]
        if np.isnan(v).all():
            continue
        c = float(np.nanmedian(v))
        octave = round((c - 100 * s) / 1200)
        out.append((s, 100 * s + 1200 * octave))
    return out


def directions(notes, counts):
    """Add (up, down) approaches per swar; repeats of the same note are not an approach."""
    for (_, a), (s, b) in zip(notes, notes[1:]):
        if b != a:
            counts.setdefault(s, [0, 0])[0 if b > a else 1] += 1


def measure(raag):
    scale = sorted(raagdb.dataset_raags([raag])[raag].scale)
    reader = json.loads(fit_reader.READER_JSON.read_text())
    params = dict(C.READ_MATCH, swar_offsets=reader["swar_offsets"])
    onsets = (reader["onset_slow"], reader["onset_fast"])
    counts = {"held notes": {}, "reader": {}}
    videos = fullaudio.cached_videos(tuple([raag]))
    for v in videos:
        ctr = fullaudio.contour(v)
        for a, b in phrases(ctr):
            seg = ctr.cents[a:b]
            directions(held_notes(seg, ctr.hop, scale), counts["held notes"])
            directions(reader_notes(seg, ctr.hop, scale, params, onsets), counts["reader"])
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
                      f"{m}: {row[m]['up']:4d} up {row[m]['down']:4d} down" for m in counts))
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
