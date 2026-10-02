"""Insight functions: what a clip says about its raag, beyond "is this phrase here".

    insights(cents, hop)   aarohi / avarohi swars and nyas swars of a clip, as swar names

**Audio only:** a clip comes with its Sa (the tonic-relative contour), never its raag, so nothing
here takes a raag or restricts notes to a scale. Raag labels may be used to *train*, never to infer.

  aarohi / avarohi   a swar is aarohi if, after it, the music moves up (practically: up >= dir_ratio x
                     down); avarohi the reverse. Judged by the *next* note, never the one before.
                     Only notes of >= dir_min_note_s count: shorter ones are kan and transit.
  nyas               the swar a breath or pause follows -- not the longest note, and not where a
                     phrase ends (P M G m G R S, sung in one breath, rests on S). A pause is an
                     unvoiced run >= pause_min_s that is also long *for the local pace*
                     (>= pause_rel x the median note length around it), or any silence >= phrase_gap_s.

Every note comes from the tuned heuristic notes (the reader fitted to notation, `fit_reader.load()`),
so only notes that "count" by that training count here.

Terms: [DATA.md § Glossary](../DATA.md#glossary).
"""

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import decode
import fit_reader
from utils import raagdb

P = C.INSIGHTS


def phrases(cents, hop, gap_s=None, min_s=None):
    """(a, b) frame bounds of voiced stretches, split at unvoiced gaps longer than `gap_s`."""
    gap_s = P["phrase_gap_s"] if gap_s is None else gap_s
    min_s = P["min_phrase_s"] if min_s is None else min_s
    voiced = ~np.isnan(np.asarray(cents, float))
    gap = max(1, int(gap_s / hop))
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


def notes(cents, hop, scale=None, reader=None):
    """Tuned heuristic notes: [(swar 0-11, pitch in cents incl. octave, t0, t1 seconds)]."""
    params, onsets = reader or fit_reader.load()
    p = dict(params, onset_cost=onsets[1] if fit_reader.density(cents, hop)
             >= fit_reader.FAST_NOTES_PER_S else onsets[0])
    seq, spans = decode.free_read(cents, hop, params=p, allowed=scale)
    out = []
    for s, (t0, t1) in zip(seq, spans):
        v = cents[int(t0 / hop):int(t1 / hop)]
        if np.isnan(v).all():
            continue
        c = float(np.nanmedian(v))
        out.append((s, 100 * s + 1200 * round((c - 100 * s) / 1200), t0, t1))
    return out


def directions(ns, counts):
    """Add [up, down] per swar, judged by the *next* note; repeats of a note are not a move."""
    dedup = [n for i, n in enumerate(ns) if i == 0 or n[1] != ns[i - 1][1]]
    for a, b in zip(dedup, dedup[1:]):
        counts.setdefault(a[0], [0, 0])[0 if b[1] > a[1] else 1] += 1


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


def final_swar(ns, rule=None):
    """The swar a pause follows: the last note before it, unless that note is a short release
    (shorter than skip_short_s -- and, with droop_only, lower than the note before)."""
    rule = {**P, **(rule or {})}
    k = len(ns) - 1
    while k > 0 and ns[k][3] - ns[k][2] < rule["skip_short_s"] and (
            not rule["droop_only"] or ns[k][1] < ns[k - 1][1]):
        k -= 1
    return ns[k][0] if ns else None


def loudness(wav, hop, n):
    """RMS level in dB per contour frame, from the clip's audio (same start as the contour)."""
    import librosa
    y, sr = librosa.load(wav, sr=None, mono=True)
    h = max(1, int(round(hop * sr)))
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=4 * h, hop_length=h)[0])
    return np.pad(db, (0, max(0, n - len(db))), mode="edge")[:n]


def read(cents, hop, reader=None):
    """The expensive part, done once per clip: [(a, b, notes)] per stretch between silences of
    >= phrase_gap_s, note times relative to the clip."""
    reader = reader or fit_reader.load()
    cents = np.asarray(cents, float)
    return [(a, b, [(s, c, t0 + a * hop, t1 + a * hop) for s, c, t0, t1 in
                    notes(cents[a:b], hop, None, reader)]) for a, b in phrases(cents, hop)]


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


def nyas_events(cents, hop, read_, p=None, loud=None):
    """[(swar, pause time s)]: the swar each pause follows. A pause is an unvoiced run that is
    >= pause_min_s, >= pause_rel x the local median note length, and -- given `loud` and
    pause_drop_db -- quieter than the 0.3 s before it by that many dB (a voice that merely tapers
    loses its pitch track without falling silent). A run reaching the clip's end is a cut, not a pause."""
    p = {**P, **(p or {})}
    cents = np.asarray(cents, float)
    voiced = ~np.isnan(cents)
    ns_all = [n for _, _, ns in read_ for n in ns]
    out, prev = [], 0.0
    for a, b, _ in read_:
        for g0, g1 in _gaps(voiced, a, b):
            if g1 >= len(cents) or g0 == 0:
                continue
            g = (g1 - g0) * hop
            t = g0 * hop
            near = [n[3] - n[2] for n in ns_all if abs((n[2] + n[3]) / 2 - t) <= p["pace_window_s"]]
            if g < p["pause_min_s"] or g < p["pause_rel"] * (float(np.median(near)) if near else 0.0):
                continue
            if loud is not None and p.get("pause_drop_db"):
                pre = loud[max(0, g0 - int(0.3 / hop)):g0]
                if not len(pre) or np.mean(pre) - np.mean(loud[g0:g1]) < p["pause_drop_db"]:
                    continue
            before = [n for n in ns_all if n[3] <= t + hop and n[2] >= prev - hop]
            if before:
                out.append((final_swar(before, p), round(t, 2)))
            prev = t
    return out


def moves(read_, p=None):
    """[up, down] per swar over a clip, counting only notes of >= dir_min_note_s."""
    p = {**P, **(p or {})}
    counts = {}
    for _, _, ns in read_:
        directions([n for n in ns if n[3] - n[2] >= p["dir_min_note_s"]], counts)
    return counts


def nyas(ends, min_count=None, min_share=None):
    """{'nyas': [...], 'counts': {swar: pauses it precedes}} from the swars before each pause."""
    min_count = P["nyas_min_count"] if min_count is None else min_count
    min_share = P["nyas_min_share"] if min_share is None else min_share
    ends = [e for e in ends if e is not None]
    counts = {s: ends.count(s) for s in sorted(set(ends))}
    n = max(len(ends), 1)
    return {"nyas": [s for s, c in counts.items() if c >= min_count and c / n >= min_share],
            "counts": counts}


def insights(cents, hop, reader=None, wav=None, direction="learned", nyas_by="learned"):
    """Both insights for one clip (tonic-relative contour; no raag), as swar names plus the counts.

    direction: "learned" (default; the frozen detector) or "rules" (threshold heuristics in
    config.INSIGHTS). nyas_by: "learned" (default; needs the clip's audio `wav`, else falls back)
    or "rules"."""
    name = raagdb.SWAR_NAMES
    cents = np.asarray(cents, float)
    r = read(cents, hop, reader)
    mv = moves(r)
    it = dict(cents=cents, hop=hop, read=r)
    if direction == "learned":
        from insights import detect
        said = detect.direction_predict(it, detect.load_choice("directions")["direction"])
        uni = {k: [s for s, v in said.items() if v == k] for k in ("aarohi", "avarohi")}
    else:
        uni = unidirectional(mv)
    if nyas_by == "learned" and wav is not None:
        from insights import detect, voice
        it.update(loud=loudness(wav, hop, len(cents)), voice=voice.voice_db(wav, hop, len(cents)))
        m = detect.load_choice("nyas")
        ev = detect.nyas_predict(it, m["nyas"], m["nyas_threshold"])
    else:
        ev = nyas_events(cents, hop, r)
    ny = nyas([e for e, _ in ev])
    return dict(
        aarohi=[name[s] for s in sorted(uni["aarohi"])], avarohi=[name[s] for s in sorted(uni["avarohi"])],
        nyas=[name[s] for s in ny["nyas"]],
        moves={name[s]: dict(up=u, down=d) for s, (u, d) in sorted(mv.items())},
        pauses_after={name[s]: c for s, c in ny["counts"].items()},
        n_pauses=len(ev), pause_times=[(name[e], t) for e, t in ev])
