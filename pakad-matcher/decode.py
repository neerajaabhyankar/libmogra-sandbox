"""Explain a span two ways, and compare: forced through a phrase, or free to use any swars.

The matcher's cost is *absolute* -- it says how well a span fits a phrase, not whether the phrase
is the best account of that span. A stretch sitting quietly on two swars fits half the phrases in
the database at cost ~0. The ratio here asks the comparative question instead:

    ratio = (cost of the best path that IS the phrase - cost of the best path that is anything)
            / frames

Both decodes use the same emissions, the same ornament/transit rules and the same minimum dwell,
so the difference is only the constraint. 0 means the phrase is as good an account as any; large
means the span is really something else and the phrase was forced onto it.
"""

import numpy as np

import _bootstrap  # noqa: F401
import config as C
from matcher import _held, _in_band

INF = np.inf


def _note_emissions(cents, p):
    """T x 12 cost of calling each frame each swar (the matcher's per-frame note cost)."""
    target = 100.0 * np.arange(12)
    d = np.abs((cents[:, None] - target[None, :] + 600.0) % 1200.0 - 600.0)
    E = np.minimum(np.maximum(0.0, d - p["free_cents"]) / p["scale_cents"], p["note_cap"])
    E[np.isnan(cents)] = p["gap_cost"]
    return E


def _orn_emission(cents, a, b, moving, p):
    """Cost of calling each frame an ornament between swars a and b."""
    e = np.where(_in_band(cents, 100.0 * a, 100.0 * b, p["kan_cents"]) & moving,
                 p["transit_cost"], p["orn_cost"])
    return np.where(np.isnan(cents), p["gap_cost"], e)


def _run(E, A, starts, ends, path=False):
    """Viterbi over a span that must be covered end to end. Best total cost, or (cost, states)."""
    T, S = E.shape
    D = np.where(starts, 0.0, INF) + E[0]
    bp = np.zeros((T, S), np.int16) if path else None
    for t in range(1, T):
        tot = D[:, None] + A
        best = np.argmin(tot, axis=0)
        if path:
            bp[t] = best
        D = tot[best, np.arange(S)] + E[t]
    end_idx = np.flatnonzero(ends)
    j = end_idx[int(np.argmin(D[ends]))]
    if not path:
        return float(D[j])
    states = [j]
    for t in range(T - 1, 0, -1):
        j = int(bp[t, j])
        states.append(j)
    return float(D[states[0]]), np.array(states[::-1])


def phrase_cost(cents, swars, hop, p, path=False, free_edges=False):
    """Best cost of a path through exactly this phrase, covering the whole span.

    `free_edges` adds a state at each end that absorbs whatever the notator is not accounting for
    -- the blip before they started, the silence after -- at the same per-frame price as an
    unexplained ornament. Without it, one forced note at the rim ruins an otherwise right reading.
    """
    n = max(1, int(round(p["min_dwell_s"] / hop)))
    K = len(swars)
    note_E = _note_emissions(cents, p)
    moving = ~_held(cents, hop, p)

    cols, kinds = [], []
    for k, s in enumerate(swars):
        for j in range(n):
            cols.append(note_E[:, s]); kinds.append(("note", k, j))
        if k < K - 1:
            cols.append(_orn_emission(cents, swars[k], swars[k + 1], moving, p))
            kinds.append(("orn", k, 0))
    if free_edges:
        edge = np.full(len(cents), p["orn_cost"])
        edge[np.isnan(cents)] = p["gap_cost"]
        cols = [edge] + cols + [edge]
        kinds = [("edge", -1, 0)] + kinds + [("edge", -1, 1)]
    E = np.stack(cols, axis=1)
    S = len(kinds)

    A = np.full((S, S), INF)
    for i, (kind, k, j) in enumerate(kinds):
        if kind == "edge":
            A[i, i] = 0.0
            if i + 1 < S:
                A[i, i + 1] = 0.0
            continue
        if kind == "orn" or j == n - 1:
            A[i, i] = 0.0                                   # hold
        if i + 1 < S:
            A[i, i + 1] = 0.0                               # advance
        if kind == "note" and j == n - 1 and i + 2 < S and kinds[i + 1][0] == "orn":
            A[i, i + 2] = 0.0                               # skip the ornament
    starts = np.array([(kind == "note" and k == 0 and j == 0) or (kind == "edge" and j == 0)
                       for kind, k, j in kinds])
    ends = np.array([(kind == "note" and k == K - 1 and j == n - 1) or (kind == "edge" and j == 1)
                     for kind, k, j in kinds])
    if not path:
        return _run(E, A, starts, ends)
    cost, states = _run(E, A, starts, ends, path=True)
    per_frame = np.array([kinds[s][1] if kinds[s][0] == "note"
                          else (-2 if kinds[s][0] == "edge" else -1) for s in states])
    return cost, per_frame


def free_cost(cents, hop, p, allowed=None):
    """Best cost of any swar sequence at all, under the same rules.

    `allowed` restricts the free model to a set of swars (e.g. the raag's scale) -- a harder
    baseline, asking "is this span better explained by *some other movement of this raag*".
    """
    n = max(1, int(round(p["min_dwell_s"] / hop)))
    note_E = _note_emissions(cents, p)
    moving = ~_held(cents, hop, p)
    swar_set = sorted(allowed) if allowed is not None else list(range(12))

    cols = [note_E[:, s] for s in swar_set for _ in range(n)]
    pairs = [(a, b) for a in swar_set for b in swar_set if a != b]
    cols += [_orn_emission(cents, a, b, moving, p) for a, b in pairs]
    E = np.stack(cols, axis=1)
    S = E.shape[1]
    last = {s: i * n + n - 1 for i, s in enumerate(swar_set)}  # final sub-state of each swar
    first = {s: i * n for i, s in enumerate(swar_set)}
    orn_at = {ab: len(swar_set) * n + i for i, ab in enumerate(pairs)}

    A = np.full((S, S), INF)
    for i, s in enumerate(swar_set):
        for j in range(n - 1):
            A[i * n + j, i * n + j + 1] = 0.0
        A[last[s], last[s]] = 0.0
    for a, b in pairs:
        o = orn_at[(a, b)]
        A[o, o] = 0.0
        A[last[a], o] = 0.0
        A[o, first[b]] = 0.0
        A[last[a], first[b]] = 0.0                          # straight to the next swar
    starts = np.zeros(S, bool)
    ends = np.zeros(S, bool)
    for s in swar_set:
        starts[first[s]] = True
        ends[last[s]] = True
    return _run(E, A, starts, ends)


def align(cents, swars, hop, params=None, free_edges=True):
    """Which frame is which swar of `swars`, over the whole span. -1 = ornament/transit.

    This is what a notation UI needs: the notator types the sequence they hear, and the model
    places it on the contour, rather than asking them to mark every note by hand.
    """
    p = {**C.MATCH, **(params or {})}
    cost, kinds = phrase_cost(cents, swars, hop, p, path=True, free_edges=free_edges)
    return kinds, cost / max(1, len(cents))


def ratio(cents, swars, hop, params=None, allowed=None, per="frame"):
    """Extra cost of insisting the span is this phrase. Lower = more likely the phrase.

    `per`: "frame" normalises by span length, "note" by the number of swars in the phrase.
    """
    p = {**C.MATCH, **(params or {})}
    if len(cents) < 2:
        return np.nan
    extra = phrase_cost(cents, swars, hop, p) - free_cost(cents, hop, p, allowed)
    return extra / (len(cents) if per == "frame" else len(swars))
