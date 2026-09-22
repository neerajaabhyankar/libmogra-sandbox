"""S1: locate a phrase in a contour by subsequence Viterbi over a left-to-right model.

    note k:  MIN_DWELL chained sub-states, the last with a self-loop  (time dilation)
    orn k:   between note k and k+1, emits anything at a flat cost     (kan swar, meend transit)
    start:   free at any frame; end: any frame in the last note        (subsequence alignment)

Costs are in config.MATCH. The DP minimises total cost; candidates are then re-scored with
duration-invariant terms (worst note's pitch misfit + ornament-time fraction), so a slow
alap rendering and a fast taan rendering of the same phrase compete on equal terms.
"""

from dataclasses import dataclass

import numpy as np

import config as C

INF = np.inf


@dataclass
class Candidate:
    t0: float
    t1: float
    cost: float          # re-scored, lower is better
    pitch_cost: float    # worst note's misfit (each note scored on its best-fitting frames)
    orn_frac: float      # fraction of the interval in ornament excursions (glides excluded)
    gap_frac: float      # fraction unvoiced
    leaps: int           # steps taken in the wrong direction / octave
    path: np.ndarray     # (n_frames,) state kind per frame: note index k, or -1 ornament
    f0: int              # first frame index
    f1: int              # last frame index (inclusive)


def _states(swars, n_dwell):
    """Per state: target cents (NaN = ornament), note index (-1 = ornament), self-loop flag."""
    target, note, loop = [], [], []
    for k, s in enumerate(swars):
        for j in range(n_dwell):
            target.append(100.0 * s); note.append(k); loop.append(j == n_dwell - 1)
        if k < len(swars) - 1:
            target.append(np.nan); note.append(-1); loop.append(True)
    return np.array(target), np.array(note), np.array(loop)


def _transitions(note, loop):
    """Dense S x S additive transition cost: 0 where allowed, inf elsewhere."""
    S = len(note)
    A = np.full((S, S), INF)
    for i in range(S):
        if loop[i]:
            A[i, i] = 0.0
        if i + 1 < S:
            A[i, i + 1] = 0.0                      # next sub-state / note -> orn / orn -> note
        if note[i] >= 0 and loop[i] and i + 2 < S and note[i + 1] == -1:
            A[i, i + 2] = 0.0                      # skip the ornament: note k -> note k+1
    return A


def _held(cents, hop, p):
    """Frames in a run of >= held_min_s where pitch moves slower than held_slope (cents/s).
    Kan swars and meend move; a held note sits. Peaks of a kan are too brief to count."""
    w = max(1, int(round(p["held_win_s"] / hop / 2)))
    slope = np.full(len(cents), np.inf)
    if len(cents) > 2 * w:
        slope[w:-w] = np.abs(cents[2 * w:] - cents[:-2 * w]) / (2 * w * hop)
    slow = np.nan_to_num(slope, nan=np.inf) < p["held_slope"]
    out = np.zeros(len(cents), bool)
    n_min = max(1, int(round(p["held_min_s"] / hop)))
    start = None
    for t, v in enumerate(np.append(slow, False)):
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start + 2 * w >= n_min:          # the window erodes w frames at each end
                out[max(0, start - w):t + w] = True
            start = None
    return out


def _in_band(cents, a, b, tol):
    """Is folded pitch within `tol` of the shortest path from swar-cents a to b?"""
    step = (b - a + 600.0) % 1200.0 - 600.0
    lo, hi = a + min(0.0, step) - tol, a + max(0.0, step) + tol
    x = (cents - lo) % 1200.0                     # position above lo, folded
    return x <= hi - lo


def _emissions(cents, target, p, hop):
    """T x S emission costs, and a per-frame 'break' mask (long unvoiced runs)."""
    voiced = ~np.isnan(cents)
    d = np.abs((cents[:, None] - target[None, :] + 600.0) % 1200.0 - 600.0)
    E = np.minimum(np.maximum(0.0, d - p["free_cents"]) / p["scale_cents"], p["note_cap"])
    moving = ~_held(cents, hop, p)
    for j in np.flatnonzero(np.isnan(target)):
        E[:, j] = np.where(_in_band(cents, target[j - 1], target[j + 1], p["kan_cents"]) & moving,
                           p["transit_cost"], p["orn_cost"])
    E[~voiced] = p["gap_cost"]
    # unvoiced runs longer than max_gap_s break matches
    brk = np.zeros(len(cents), bool)
    run_start = None
    for t, v in enumerate(np.append(voiced, True)):
        if not v and run_start is None:
            run_start = t
        elif v and run_start is not None:
            if (t - run_start) * hop > p["max_gap_s"]:
                brk[run_start:t] = True
            run_start = None
    E[brk] = INF
    return E + p["step_eps"], brk


def _viterbi(E, brk, A):
    T, S = E.shape
    D = np.full(S, INF)
    ends = np.full(T, INF)
    bp = np.zeros((T, S), np.int16)
    for t in range(T):
        tot = D[:, None] + A                       # S x S
        bp[t] = np.argmin(tot, axis=0)
        D = tot[bp[t], np.arange(S)]
        if not brk[t] and D[0] > 0.0:              # free start
            D[0], bp[t, 0] = 0.0, -1
        D = D + E[t]
        ends[t] = D[-1]
    return ends, bp


def _backtrack(bp, t_end, S):
    states, s, t = [], S - 1, t_end
    while t >= 0:
        states.append(s)
        prev = bp[t, s]
        if prev < 0:
            break
        s, t = int(prev), t - 1
    return t, np.array(states[::-1])


def _rescore(cents, states, note, target, t0, p, hop):
    seg = cents[t0:t0 + len(states)]
    kinds = note[states]
    voiced = ~np.isnan(seg)
    per_note = []
    for k in range(kinds.max() + 1):
        m = (kinds == k) & voiced
        if m.any():
            d = np.abs((seg[m] - target[states[m]] + 600.0) % 1200.0 - 600.0)
            fc = np.sort(np.minimum(np.maximum(0.0, d - p["free_cents"]) / p["scale_cents"], p["note_cap"]))
            per_note.append(np.mean(fc[:max(1, int(np.ceil(len(fc) * p["note_trim"])))]))
        else:
            per_note.append(p["note_cap"])
    pitch = float(np.max(per_note))
    held = _held(seg, hop, p) & (kinds == -1)
    orn = float(np.mean(_excursion(seg, kinds, p["kan_cents"]) | held))
    gap = float(np.mean(~voiced))
    leaps = _wrong_steps(seg, kinds, target, states)
    return pitch, orn, gap, leaps, pitch + p["orn_weight"] * orn + gap + p["leap_penalty"] * leaps


def _wrong_steps(seg, kinds, target, states):
    """Steps between consecutive notes that go the wrong way or jump an octave.
    Intended step = the shortest one between the two swars (octave folding hides direction)."""
    n = 0
    for k in range(kinds.max()):
        a, b = seg[kinds == k], seg[kinds == k + 1]
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        ta, tb = target[states[kinds == k][0]], target[states[kinds == k + 1][0]]
        intended = (tb - ta + 600.0) % 1200.0 - 600.0
        actual = np.nanmedian(b) - np.nanmedian(a)
        err = abs(actual - intended)
        if abs(abs(intended) - 600.0) < 1e-6:      # a tritone is equally short either way
            err = min(err, abs(actual - (intended + 1200.0)))
        n += err > 600.0
    return int(n)


def _excursion(seg, kinds, tol):
    """Ornament frames that leave the pitch range spanned by their two neighbouring notes.
    A glide from one note to the next, or a kan overshooting it by up to `tol`, costs nothing.
    (Held notes inside an ornament slot are charged separately, in `_rescore`.)"""
    out = np.zeros(len(seg), bool)
    for k in range(kinds.max()):
        m = kinds == -1
        idx = np.flatnonzero(m & (np.arange(len(kinds)) > np.flatnonzero(kinds == k).max())
                             & (np.arange(len(kinds)) < np.flatnonzero(kinds == k + 1).min()))
        if not len(idx):
            continue
        na, nb = seg[kinds == k], seg[kinds == k + 1]
        if np.isnan(na).all() or np.isnan(nb).all():
            continue
        a, b, x = np.nanmedian(na), np.nanmedian(nb), seg[idx]
        out[idx] = ~np.isnan(x) & ((x < min(a, b) - tol) | (x > max(a, b) + tol))
    return out


def _extend(cents, t0, t1, first, last, p, hop, max_s=1.0):
    """Grow the interval over the rest of a held first/last note (the DP trims holds)."""
    lim = int(max_s / hop)
    near = lambda t, s: not np.isnan(cents[t]) and abs((cents[t] - 100 * s + 600) % 1200 - 600) <= p["free_cents"]
    a = t0
    while a > 0 and t0 - a < lim and near(a - 1, first):
        a -= 1
    b = t1
    while b < len(cents) - 1 and b - t1 < lim and near(b + 1, last):
        b += 1
    return a, b


def match(contour, swars, top_k=C.TOP_K, params=None):
    """Top-k non-overlapping candidates for `swars` (0..11, collapsed) in `contour`."""
    p = {**C.MATCH, **(params or {})}
    hop = contour.hop
    n_dwell = max(1, int(round(p["min_dwell_s"] / hop)))
    target, note, loop = _states(swars, n_dwell)
    E, brk = _emissions(contour.cents, target, p, hop)
    ends, bp = _viterbi(E, brk, _transitions(note, loop))

    out = []
    for t_end in np.argsort(ends):
        if not np.isfinite(ends[t_end]) or len(out) >= C.CANDIDATE_POOL:
            break
        t_first, states = _backtrack(bp, int(t_end), len(note))
        t0 = int(t_end) - len(states) + 1
        if any(_iou((t0, t_end), (c.f0, c.f1)) > C.NMS_IOU for c in out):
            continue
        pitch, orn, gap, leaps, cost = _rescore(contour.cents, states, note, target, t0, p, hop)
        a, b = _extend(contour.cents, t0, int(t_end), swars[0], swars[-1], p, hop)
        path = np.concatenate([np.full(t0 - a, 0), note[states], np.full(b - int(t_end), len(swars) - 1)])
        out.append(Candidate(a * hop, (b + 1) * hop, cost, pitch, orn, gap, leaps, path, a, b))
    return sorted(out, key=lambda c: c.cost)[:top_k]


def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    return inter / (max(a[1], b[1]) - min(a[0], b[0]) + 1)
