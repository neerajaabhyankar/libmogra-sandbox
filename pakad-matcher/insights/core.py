"""Insight functions: what a clip says about its raag, beyond "is this phrase here".

    insights(cents, hop, wav=None)   aarohi / avarohi swars and nyas swars of a clip, as swar names

**Audio only:** a clip comes with its Sa (Neeraja, 2026-10-03: Sa is given), never its raag. Nothing
here takes a raag, a scale or anything from the raag DB. Raag labels may train; they never infer.

  aarohi / avarohi   after X only higher notes follow (aarohi) / only lower (avarohi), judged by the
                     next *sung* note within a breath (notes.directions). Threshold heuristic:
                     up >= dir_ratio x down, with >= dir_min_count ups (or the mirror).
  nyas               the swar a breath or pause follows -- not the longest note, not a phrase end
                     (P M G m G R S in one breath rests on S); a short final note can be the nyas.
                     Threshold heuristic for a pause: an unvoiced run >= pause_min_s that is also
                     >= pause_rel x the local median note length, or any run >= pause_abs_s.

Which method answers each question -- the threshold heuristics here or the learned detectors
(detect.py) -- is the frozen choice in results/insights/choice.json, made without the test clips.
Terms: [DATA.md § Glossary](../DATA.md#glossary).
"""

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import notes
from utils import raagdb

P = C.INSIGHTS


def unidirectional(counts, ratio=None, min_count=None):
    """{'aarohi': [...], 'avarohi': [...]}: up >= ratio x down (with up >= min_count), and vice versa."""
    ratio = P["dir_ratio"] if ratio is None else ratio
    min_count = P["dir_min_count"] if min_count is None else min_count
    out = {"aarohi": [], "avarohi": []}
    for s, (up, down) in sorted(counts.items()):
        if up >= min_count and up >= ratio * down:
            out["aarohi"].append(s)
        elif down >= min_count and down >= ratio * up:
            out["avarohi"].append(s)
    return out


def moves(read_):
    """[up, down] per swar over a clip (notes.directions within each breath)."""
    counts = {}
    for _, _, ns in read_:
        notes.directions(ns, counts)
    return counts


def loudness(wav, hop, n):
    """RMS level in dB per contour frame, from the clip's audio (same start as the contour)."""
    import librosa
    y, sr = librosa.load(wav, sr=None, mono=True)
    h = max(1, int(round(hop * sr)))
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=4 * h, hop_length=h)[0])
    return np.pad(db, (0, max(0, n - len(db))), mode="edge")[:n]


def _gaps(voiced, a, b):
    """(start, end) frames of unvoiced runs after frame a, up to the next voiced frame past b."""
    out, t = [], a
    while t < len(voiced):
        if voiced[t]:
            if t >= b:
                break
            t += 1; continue
        u = t
        while u < len(voiced) and not voiced[u]:
            u += 1
        out.append((t, u))
        if u >= b:
            break
        t = u
    return out


def nyas_events(cents, hop, read_, p=None):
    """[(swar, pause time s)]: the last note before each pause (threshold heuristic). A run that
    reaches the clip's end is a cut, not a pause."""
    p = {**P, **(p or {})}
    cents = np.asarray(cents, float)
    voiced = ~np.isnan(cents)
    ns_all = [n for _, _, ns in read_ for n in ns]
    out, prev = [], 0.0
    for a, b, _ in read_:
        for g0, g1 in _gaps(voiced, a, b):
            if g1 >= len(cents) or g0 == 0:
                continue
            g, t = (g1 - g0) * hop, g0 * hop
            near = [n[3] - n[2] for n in ns_all if abs((n[2] + n[3]) / 2 - t) <= p["pace_window_s"]]
            pace = float(np.median(near)) if near else 0.0
            if not (g >= p["pause_abs_s"] or (g >= p["pause_min_s"] and g >= p["pause_rel"] * pace)):
                continue
            before = [n for n in ns_all if n[3] <= t + hop and n[2] >= prev - hop]
            if before:
                out.append((before[-1][0], round(t, 2)))
            prev = t
    return out


def nyas(ends, min_count=None, min_share=None):
    """{'nyas': [...], 'counts': {swar: pauses it precedes}} from the swars before each pause."""
    min_count = P["nyas_min_count"] if min_count is None else min_count
    min_share = P["nyas_min_share"] if min_share is None else min_share
    ends = [e for e in ends if e is not None]
    counts = {s: ends.count(s) for s in sorted(set(ends))}
    n = max(len(ends), 1)
    return {"nyas": [s for s, c in counts.items() if c >= min_count and c / n >= min_share],
            "counts": counts}


def insights(cents, hop, reader=None, wav=None):
    """Both insights for one clip (Sa-relative contour; no raag), with the counts behind them.
    Uses the frozen choice per question; the learned nyas detector needs the clip's audio (`wav`)
    -- without it the nyas heuristic is used, and `nyas_method` says so."""
    from insights import detect
    cents = np.asarray(cents, float)
    it = dict(cents=cents, hop=hop, read=notes.read(cents, hop, reader))
    if wav is not None:
        from insights import voice
        it.update(loud=loudness(wav, hop, len(cents)), voice=voice.voice_db(wav, hop, len(cents)))
    said = detect.frozen("directions")(it)
    nyas_by = detect.frozen("nyas", audio=wav is not None)
    ev = nyas_by(it)
    ny = nyas([e for e, _ in ev])
    name, mv = raagdb.SWAR_NAMES, moves(it["read"])
    return dict(
        aarohi=[name[s] for s, k in sorted(said.items()) if k == "aarohi"],
        avarohi=[name[s] for s, k in sorted(said.items()) if k == "avarohi"],
        nyas=[name[s] for s in ny["nyas"]], nyas_method=nyas_by.method,
        moves={name[s]: dict(up=u, down=d) for s, (u, d) in sorted(mv.items())},
        pauses_after={name[s]: c for s, c in ny["counts"].items()},
        n_pauses=len(ev), pause_times=[(name[e], t) for e, t in ev])
