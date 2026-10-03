"""Every score and interval reported in plan.md, in one place.

    edit_ops(a, b)                     substitutions, deletions, insertions between swar sequences
    per_samooha_auc(score, items)      does a 'yes' outrank a 'no', within each samooha?
    precision_at(score, items, k)      share of yeses in each samooha's top k (ties shared fairly)
    bootstrap(stat, groups)            95% interval by resampling whole groups (samoohas, clips, swars)

Before 2026-10-03 the intervals in plan.md came from throwaway scripts that were never saved;
from then on they come from `bootstrap` here, called by the script that reports the number.
Terms: [DATA.md § Glossary](DATA.md#glossary).
"""

import numpy as np

BOOT_N = 2000
BOOT_SEED = 0


def edit_ops(a, b):
    """Levenshtein between swar sequences; returns (substitutions, deletions, insertions)."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, sub, dele, ins = n, m, 0, 0, 0
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            sub += a[i - 1] != b[j - 1]; i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            dele += 1; i -= 1                      # in the notation, missing from the reading
        else:
            ins += 1; j -= 1                       # in the reading, not in the notation
    return sub, dele, ins


def per_samooha_auc(score, items):
    """Does a 'yes' outrank a 'no' within each samooha? Higher score = more likely the phrase.
    Samoohas with only yeses (or only nos) have no AUC and are left out."""
    score = np.asarray(score)
    pid = np.array([it["pid"] for it in items])
    y = np.array([it["y"] for it in items])
    aucs = []
    for p in sorted(set(pid)):
        m = pid == p
        a, b = score[m][y[m] == 1], score[m][y[m] == 0]
        if len(a) and len(b):
            aucs.append(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))
    return float(np.mean(aucs)) if aucs else np.nan


def _top_k_yes(s, y, k):
    """Expected yes-share of the top k when tied scores are taken in random order."""
    k = min(k, len(s))
    order = np.sort(s)[::-1]
    edge = order[k - 1]
    above = s > edge
    tied = s == edge
    need = k - above.sum()
    return (y[above].sum() + need * y[tied].mean()) / k


def precision_at(score, items, k):
    """Mean over samoohas of the yes-share among the k best-scored candidates. Ties at the k-th
    place are shared fairly (expected value over tie orders); before 2026-10-03 they were broken by
    pool order, which made P@1 of a method with many ties depend on how the pool was listed."""
    score = np.asarray(score, float)
    pid = np.array([it["pid"] for it in items])
    y = np.array([it["y"] for it in items], float)
    return float(np.mean([_top_k_yes(score[pid == p], y[pid == p], k) for p in sorted(set(pid))]))


def bootstrap(stat, groups, n=BOOT_N, seed=BOOT_SEED):
    """(estimate, lo, hi): `stat(list of groups)` on the data, and its 95% interval when whole
    groups are resampled with replacement. Resamples where `stat` is undefined (NaN) are skipped."""
    rng = np.random.default_rng(seed)
    groups = list(groups)
    vals = []
    for _ in range(n):
        v = stat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if v is not None and not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(stat(groups)), float(lo), float(hi)


def by_group(items, key):
    """Items grouped for `bootstrap`: [[items of group 1], [items of group 2], ...]."""
    out = {}
    for it in items:
        out.setdefault(it[key], []).append(it)
    return list(out.values())
