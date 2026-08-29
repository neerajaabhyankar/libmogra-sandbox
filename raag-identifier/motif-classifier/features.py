"""Per-clip sequence statistics, computed once and reused across every method.

All three methods score a clip against 50 raags at each of 12 tonic rotations. Doing that
from the raw swar string every time makes a hyperparameter sweep unaffordable, so the
expensive part — turning a swar string into n-gram / skip-gram counts and a duration
histogram — happens here, once per clip, at rotation 0. Every other rotation is a
*permutation* of those same counts, so the 12-way tonic search costs nothing extra.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np


def ngrams_with_skips(swars, n, max_skip=0, skip_decay=0.5):
    """Counts of length-`n` swar patterns, allowing up to `max_skip` notes to be stepped over.

    Skips are what make this tolerant of the thing that breaks exact matching: a kan-swar,
    an ornament note, or a tracker glitch sitting in the middle of an otherwise-textbook
    phrase. A pattern found only by skipping k notes is discounted by `skip_decay ** k`.
    """
    counts = defaultdict(float)
    L = len(swars)
    for i in range(L):
        for w in range(n, min(n + max_skip, L - i) + 1):
            window = swars[i : i + w]
            weight = skip_decay ** (w - n)
            if w == n:
                counts[tuple(window)] += weight
                continue
            # keep the window's endpoints, choose the rest in order
            for mid in combinations(range(1, w - 1), n - 2):
                g = (window[0],) + tuple(window[m] for m in mid) + (window[-1],)
                counts[g] += weight
    return counts


def soft_assign(cents, sigma, offsets=None):
    """Continuous cents -> a (N, 12) soft membership over swars, instead of rounding.

    Rounding to the nearest of 12 bins is the single lossiest step in this pipeline: a komal
    rishabh sung at 175 cents (which is where the Bhairav family actually puts it) rounds to
    shuddha R and the phrase is destroyed. A note at 175 cents is not "R", it is 0.75 r and
    0.25 R, and that is what this returns.

    `sigma` is the kernel width in cents (sigma -> 0 recovers hard rounding). `offsets` is an
    optional per-swar tuning offset in cents, so bin centres sit where the corpus actually
    sings them rather than on the equal-tempered grid. Distances are circular over the octave.
    """
    cents = np.asarray(cents, dtype=float).reshape(-1, 1)
    centres = np.arange(12, dtype=float) * 100.0
    if offsets is not None:
        centres = centres + np.asarray(offsets, dtype=float)
    d = np.abs(cents - centres[None, :])
    d = np.minimum(d, 1200.0 - d)  # circular
    w = np.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)
    tot = w.sum(axis=1, keepdims=True)
    return np.divide(w, tot, out=np.full_like(w, 1.0 / 12.0), where=tot > 0)


class ClipFeatures:
    """Rotation-0 statistics for one clip, plus cheap rotation to any of the 12 tonics.

    With `soft_sigma > 0` every count below is accumulated over a *soft* swar membership
    (see `soft_assign`) rather than a hard bin, so a note between two swars contributes to
    both. Every consumer sees the same array shapes, so M3/M4/M7 pick this up unchanged.
    """

    __slots__ = ("clip", "unigram_dur", "unigram_count", "bigram", "bigram_dur", "bigram_skip",
                 "trigram", "ngrams", "n_notes", "soft", "reg_dur")

    def __init__(self, clip, n_min=2, n_max=4, max_skip=1, skip_decay=0.5,
                 soft_sigma=0.0, tuning_offsets=None):
        self.clip = clip
        s, d = clip.swars, clip.durs
        self.n_notes = len(s)

        if soft_sigma > 0 and getattr(clip, "cents", None):
            self.soft = soft_assign(clip.cents, soft_sigma, tuning_offsets)
        else:
            self.soft = np.zeros((len(s), 12))
            for i, sw in enumerate(s):
                self.soft[i, sw] = 1.0

        self.unigram_dur = (self.soft * np.asarray(d, dtype=float)[:, None]).sum(axis=0) \
            if len(s) else np.zeros(12)
        self.unigram_count = self.soft.sum(axis=0) if len(s) else np.zeros(12)

        # (3, 12) duration mass by register x swar — mandra / madhya / taar. Every other
        # feature here is octave-folded, which throws away the only thing separating some
        # pairs (Deshkar and Bhoopali share a scale and differ in where the weight sits).
        # Register is only meaningful once Sa is right, so this stayed unused until v1's
        # annotated tonic made "one octave above Sa" a fact rather than an estimate.
        self.reg_dur = np.zeros((3, 12))
        if len(s):
            oct_idx = np.clip(np.asarray(clip.octaves, dtype=int), -1, 1) + 1
            dur = np.asarray(d, dtype=float)
            for r in range(3):
                m = oct_idx == r
                if m.any():
                    self.reg_dur[r] = (self.soft[m] * dur[m, None]).sum(axis=0)

        # three views of the same transitions, because the noise here is specific:
        #   bigram      plain counts
        #   bigram_dur  weighted by sqrt of the shorter note — a transition between two
        #               sustained notes is real melodic movement; one between two 0.1 s
        #               blips is usually the tracker stuttering
        #   bigram_skip pairs one note apart, so a spurious note inserted mid-phrase does
        #               not destroy the S->R evidence around it
        W = self.soft
        if len(s) >= 2:
            self.bigram = W[:-1].T @ W[1:]
            wt = np.sqrt(np.minimum(np.asarray(d[:-1]), np.asarray(d[1:])))
            self.bigram_dur = (W[:-1] * wt[:, None]).T @ W[1:]
        else:
            self.bigram = np.zeros((12, 12))
            self.bigram_dur = np.zeros((12, 12))
        self.bigram_skip = W[:-2].T @ W[2:] if len(s) >= 3 else np.zeros((12, 12))

        self.trigram = defaultdict(float)
        for a, b, c in zip(s, s[1:], s[2:]):
            self.trigram[(a, b, c)] += 1.0

        self.ngrams = {}
        for n in range(n_min, n_max + 1):
            self.ngrams[n] = dict(ngrams_with_skips(s, n, max_skip=max_skip, skip_decay=skip_decay))

    # -- rotations: shifting the tonic by k semitones just relabels the swars ------------

    def rot_unigram_dur(self, k):
        return np.roll(self.unigram_dur, k)

    def rot_unigram_count(self, k):
        return np.roll(self.unigram_count, k)

    def rot_bigram(self, k, which="bigram"):
        m = getattr(self, which)
        return np.roll(np.roll(m, k, axis=0), k, axis=1)

    def rot_ngrams(self, n, k):
        if k == 0:
            return self.ngrams[n]
        return {tuple((x + k) % 12 for x in g): v for g, v in self.ngrams[n].items()}

    def rot_swars(self, k):
        return [(x + k) % 12 for x in self.clip.swars]

    def rot_reg_dur(self, k):
        """Register x swar mass, pitch-class axis rolled.

        Only the swar axis rolls: a real tonic change would also move notes across octave
        boundaries. Harmless where this is used (M10 runs with the annotated tonic and
        `shift_mode="none"`), but it makes the rotated view an approximation, not a fact.
        """
        return np.roll(self.reg_dur, k, axis=1)

    def rot_soft(self, k):
        """(N, 12) soft memberships, rolled — soft evidence for the channel HMM."""
        return np.roll(self.soft, k, axis=1)


def build_features(clips, **kw):
    return [ClipFeatures(c, **kw) for c in clips]


class MultiFeatures:
    """One clip's features under several trackers, e.g. {"tony": ..., "crepe": ...}.

    Delegates every attribute to the primary tracker, so a method written against a plain
    `ClipFeatures` keeps working unchanged; methods that want a second opinion reach into
    `.by_tracker` explicitly.
    """

    def __init__(self, by_tracker, primary):
        self.by_tracker = by_tracker
        self.primary = primary

    def __getattr__(self, name):
        return getattr(self.by_tracker[self.primary], name)


def align_trackers(feat_sets, primary):
    """feat_sets: {tracker: [ClipFeatures]}. Returns [MultiFeatures] over clips present in all."""
    by_id = {t: {f.clip.clip_id: f for f in fs} for t, fs in feat_sets.items()}
    common = set.intersection(*(set(d) for d in by_id.values()))
    return [
        MultiFeatures({t: by_id[t][cid] for t in by_id}, primary)
        for cid in sorted(common, key=lambda c: (by_id[primary][c].clip.raag, c))
    ]


def estimate_tuning_offsets(clips, max_shift=45.0, scales=None, max_dist=90.0):
    """Where each swar is *actually* sung in this corpus, relative to the 12-TET grid.

    Duration-weighted circular mean of each note's residual from its swar's tempered
    position. Komal swars in the Bhairav/Todi families are famously sung sharp of equal
    temperament; this measures it rather than assuming it.

    The subtlety is which swar a note counts towards. Assigning to the *nearest of 12* bins
    makes the estimate collapse: any note more than 50 cents sharp is booked to the next
    swar instead, so the measured residuals are shrunk toward zero by construction (it
    returns +/-3 cents on this corpus, which is an artefact, not a finding). Passing
    `scales={raag: set_of_swars}` assigns each note to the nearest swar **the true raag
    actually contains**, which is the assignment that can see a komal rishabh sung at 175
    cents for what it is. Notes further than `max_dist` from any in-scale swar are dropped
    as transcription noise.

    Estimated on TRAIN clips only. Twelve global numbers, nothing raag-specific.
    """
    num = np.zeros(12, dtype=complex)
    den = np.zeros(12)
    for c in clips:
        if not getattr(c, "cents", None):
            continue
        allowed = None
        if scales is not None:
            allowed = sorted(scales.get(c.raag, range(12)))
            if not allowed:
                allowed = list(range(12))
        for sw, du, ct in zip(c.swars, c.durs, c.cents):
            if allowed is None:
                target, resid = sw, ct - 100.0 * sw
            else:
                ds = [(abs((ct - 100.0 * j + 600.0) % 1200.0 - 600.0), j) for j in allowed]
                dist, target = min(ds)
                if dist > max_dist:
                    continue
                resid = ct - 100.0 * target
            resid = (resid + 600.0) % 1200.0 - 600.0
            num[target] += du * np.exp(1j * 2 * np.pi * resid / 1200.0)
            den[target] += du
    off = np.zeros(12)
    for j in range(12):
        if den[j] > 0:
            off[j] = np.angle(num[j]) / (2 * np.pi) * 1200.0
    return np.clip(off, -max_shift, max_shift)
