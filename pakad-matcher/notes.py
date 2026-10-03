"""Notes from a pitch track: one definition, shared by test2 (unidir.py), the insights and the reader.

    breath_spans(cents, hop)   voiced stretches between breaths ("breath spans")
    onset_cost(cents, hop)     the reader's note-start price for this stretch's tempo
    notes(cents, hop)          the tuned heuristic notes of one stretch: (swar, pitch, t0, t1)
    read(cents, hop)           every stretch of a clip, read once, with clip-relative times
    directions(notes, counts)  [up, down] after each swar, by the next sung note

The "next note" rule is Neeraja's (2026-10-03): what decides aarohi/avarohi is the next *sung*
note after X -- kan and pass-through notes (shorter than kan_max_s) are skipped -- and direction is
never judged across a breath. Before that date test2 and the insights used different rules
(test2: any note, breaths of 0.35 s; insights: notes of >= 0.08 s, breaths of 0.25 s).

Audio only: nothing here takes a raag or a scale. Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import decode

N = C.NOTES


def breath_spans(cents, hop, breath_s=None, min_s=None):
    """(a, b) frame bounds of voiced stretches, split at unvoiced runs longer than `breath_s`."""
    breath_s = N["breath_s"] if breath_s is None else breath_s
    min_s = N["min_phrase_s"] if min_s is None else min_s
    voiced = ~np.isnan(np.asarray(cents, float))
    gap = max(1, int(breath_s / hop))
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
    return [(a, b) for a, b in out if (b - a) * hop >= min_s]


def onset_cost(cents, hop, params, onsets):
    """The reader's onset cost for this stretch: slow or fast, by its held-note density."""
    import fit_reader
    fast = fit_reader.density(cents, hop, params) >= fit_reader.FAST_NOTES_PER_S
    return onsets[1] if fast else onsets[0]


def notes(cents, hop, reader=None):
    """Tuned heuristic notes of one stretch: [(swar 0-11, pitch in cents incl. octave, t0, t1 s)]."""
    import fit_reader
    params, onsets = reader or fit_reader.load()
    cents = np.asarray(cents, float)
    p = dict(params, onset_cost=onset_cost(cents, hop, params, onsets))
    seq, spans = decode.free_read(cents, hop, params=p)
    out = []
    for s, (t0, t1) in zip(seq, spans):
        v = cents[int(t0 / hop):int(t1 / hop)]
        if np.isnan(v).all():
            continue
        c = float(np.nanmedian(v))
        out.append((s, 100 * s + 1200 * round((c - 100 * s) / 1200), t0, t1))
    return out


def read(cents, hop, reader=None):
    """Every stretch between breaths, read once: [(a, b, notes)], note times relative to the clip."""
    import fit_reader
    reader = reader or fit_reader.load()
    cents = np.asarray(cents, float)
    return [(a, b, [(s, c, t0 + a * hop, t1 + a * hop) for s, c, t0, t1 in notes(cents[a:b], hop, reader)])
            for a, b in breath_spans(cents, hop)]


def directions(ns, counts, kan_max_s=None):
    """Add [up, down] per swar to `counts`, judged by the next sung note within one stretch.
    Notes shorter than kan_max_s (kan, pass-through) are skipped; a repeated note is not a move.
    `ns` holds (swar, pitch, ...) -- with times (t0, t1) when kan are to be skipped."""
    kan = N["kan_max_s"] if kan_max_s is None else kan_max_s
    sung = [n for n in ns if len(n) < 4 or n[3] - n[2] >= kan]
    dedup = [n for i, n in enumerate(sung) if i == 0 or n[1] != sung[i - 1][1]]
    for a, b in zip(dedup, dedup[1:]):
        counts.setdefault(a[0], [0, 0])[0 if b[1] > a[1] else 1] += 1
    return counts
