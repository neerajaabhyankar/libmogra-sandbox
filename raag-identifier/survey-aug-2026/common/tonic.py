"""Sa arithmetic.

Every raag is defined *relative to its tonic*. The dataset gives us `tonic_hz` per video
(hand-annotated in v1.1), and this module turns that number into the three things a model
can actually consume:

    shift_ratio(tonic)   a multiplicative frequency factor that moves Sa onto a fixed
                         reference -- apply it to a waveform and the model never has to
                         learn the transposition (see audio.tonic_normalise)
    anchor_fmin(tonic)   a CQT `fmin` equal to Sa itself, so bin 0 *is* Sa, exactly, with
                         no rounding -- the tonic-invariant representation, for free
    conditioning(tonic)  a small feature vector for a model that keeps its audio unchanged
                         and is told the tonic instead (FiLM)

**Octave folding** runs through all of it. A raag is invariant to which octave Sa sits in,
and the annotated tonics span 101-289 Hz -- an octave and a half. Folding keeps the required
shift inside +/-6 semitones (a resample ratio in [0.707, 1.414]), which bounds the tempo
distortion that the resample trick necessarily introduces. It also matches the convention in
../motif-classifier, where everything is scored mod 12.
"""

import numpy as np

#: The reference Sa that normalised audio is moved to. D3 = 146.83 Hz, chosen because the
#: dataset's median annotated tonic is 147.0 Hz -- so about half the corpus shifts up and
#: half down, and the mean absolute distortion is as small as one fixed reference allows.
REF_TONIC_HZ = 146.83


def cents(f_hz, ref_hz):
    """Interval in cents. Vectorised."""
    return 1200.0 * np.log2(np.asarray(f_hz, dtype=float) / ref_hz)


def fold_cents(c):
    """Any interval -> the equivalent one in [-600, 600) cents. Octave equivalence."""
    return ((np.asarray(c, dtype=float) + 600.0) % 1200.0) - 600.0


def shift_ratio(tonic_hz, ref_hz=REF_TONIC_HZ):
    """Frequency factor that puts this clip's Sa on `ref_hz`, octave-folded.

    Returns a float in [2**-0.5, 2**0.5). Multiply the signal's frequencies by it -- i.e.
    play the audio `ratio` times faster -- and Sa lands on the reference (up to octaves).
    """
    return float(2.0 ** (fold_cents(cents(ref_hz, tonic_hz)) / 1200.0))


def canonical_tonic(tonic_hz, lo_hz=110.0):
    """The tonic folded into the fixed octave [lo_hz, 2*lo_hz). Used to anchor the CQT.

    Two clips a fifth apart in absolute pitch but with the same raag get CQTs whose bin 0
    is Sa in both -- which is the entire point -- while the *absolute* frequency band the
    CQT covers still varies by at most an octave across the corpus.
    """
    f = float(tonic_hz)
    while f < lo_hz:
        f *= 2.0
    while f >= 2.0 * lo_hz:
        f /= 2.0
    return f


def anchor_fmin(tonic_hz, octaves_below=1, lo_hz=110.0):
    """CQT `fmin` such that bin 0 is Sa, `octaves_below` octaves under the canonical Sa."""
    return canonical_tonic(tonic_hz, lo_hz) / (2.0 ** octaves_below)


def semitone_class(tonic_hz, ref_hz=REF_TONIC_HZ):
    """Which of 12 semitone classes the tonic falls in, relative to the reference."""
    return int(np.round(fold_cents(cents(tonic_hz, ref_hz)) / 100.0)) % 12


def conditioning(tonic_hz, ref_hz=REF_TONIC_HZ):
    """A 15-d feature vector describing the tonic, for a model that is *told* the tonic
    rather than having it normalised away.

        [0]     log2(tonic / ref)            absolute register, signed, ~[-0.6, 1.0]
        [1:3]   cos/sin of the folded cents  the pitch class as a continuous circle, so
                                             99 and 101 cents are near each other and
                                             1199 and 1 cents are too
        [3:15]  one-hot of the semitone class

    The circular pair matters more than the one-hot: a network that only sees the one-hot
    has to learn 12 unrelated embeddings from 1810 clips.
    """
    tonic_hz = float(tonic_hz)
    theta = 2.0 * np.pi * (fold_cents(cents(tonic_hz, ref_hz)) / 1200.0)
    v = np.zeros(15, dtype=np.float32)
    v[0] = np.log2(tonic_hz / ref_hz)
    v[1] = np.cos(theta)
    v[2] = np.sin(theta)
    v[3 + semitone_class(tonic_hz, ref_hz)] = 1.0
    return v


CONDITIONING_DIM = 15


def shuffled_tonics(clips, seed=0):
    """{video: tonic_hz} with the tonics permuted between videos.

    The control that catches plumbing bugs. Any experiment claiming the tonic helps must be
    re-run with these; if the number does not move, the tonic was never reaching the model
    and the "gain" is something else. (../motif-classifier's plan.md has the same rule:
    a tonic-invariant result means a bug.)
    """
    videos = sorted({c.video for c in clips})
    tonic_of = {c.video: c.tonic_hz for c in clips}
    rng = np.random.default_rng(seed)
    perm = list(videos)
    rng.shuffle(perm)
    return {v: tonic_of[p] for v, p in zip(videos, perm)}


if __name__ == "__main__":
    from common.data import load_clips

    clips = load_clips()
    t = np.array([c.tonic_hz for c in clips])
    r = np.array([shift_ratio(x) for x in t])
    print(f"tonics: {t.min():.1f}-{t.max():.1f} Hz, median {np.median(t):.1f}")
    print(f"shift ratios: {r.min():.3f}-{r.max():.3f}  (bounded by 0.707-1.414)")
    print(f"  -> tempo distortion at worst {max(r.max(), 1/r.min()):.2f}x")
    for hz in (101.0, 146.83, 220.0, 289.1):
        print(f"  {hz:7.2f} Hz -> ratio {shift_ratio(hz):.4f}, "
              f"class {semitone_class(hz):2d}, cqt fmin {anchor_fmin(hz):.2f} Hz")
    # the check that matters: after shifting, every Sa is the reference, mod octaves
    got = np.array([hz * shift_ratio(hz) for hz in t])
    err = np.abs(fold_cents(cents(got, REF_TONIC_HZ)))
    print(f"residual Sa error after normalisation: max {err.max():.6f} cents (must be ~0)")
