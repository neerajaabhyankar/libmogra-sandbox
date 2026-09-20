"""The pitch pathway is factored out, and usable without the classifier.

    pytest tests/test_pitch.py
    python  tests/test_pitch.py          # same checks, no pytest needed

Nothing here loads weights, touches the dataset, or goes near the network — the audio is
synthesised in this file. That is the point: if the only way to get a pitch track out of
this model were to build a `RaagIdentifier`, the swar histogram, the insights and anything
else that reads melody would each have to run the tracker again.

The tracker is Essentia's Melodia. Every check below is written against the *pathway*, not
against Melodia, so this file is what a version on a different tracker should still pass.

Self-contained on purpose. This directory is uploaded to the Hub, so the tests have to pass
for someone who downloaded the model and has none of the surrounding repository.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raag_fusion import melody_branch, pitch          # noqa: E402

TONIC = 146.83
TONE_HZ = 220.0
OTHER_SR = 16000            # a rate unlike the tracker's, for the resampling check
CENTS_TOLERANCE = 35.0      # trackers quantise: Melodia to a 10-cent grid, CREPE to 20


def tone(hz=TONE_HZ, seconds=3.0, sr=pitch.SR):
    """A steady note with a few partials — something any pitch tracker will follow."""
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * hz * t
    y = sum(a * np.sin(k * phase) for k, a in enumerate((1.0, 0.5, 0.25, 0.12), 1))
    return (y / np.max(np.abs(y))).astype(np.float32)


def synthetic_track(n=4000, seed=0, spread=0.5):
    """f0/voiced arrays without running the tracker, for the histogram tests."""
    rng = np.random.default_rng(seed)
    return TONIC * 2 ** rng.normal(0, spread, n), rng.random(n) < 0.8


# ---------------------------------------------------------------- the track itself

def test_pitch_is_its_own_module():
    assert callable(pitch.track)
    assert callable(pitch.from_audio)
    assert hasattr(pitch, "PitchTrack")
    # Tracker-neutral constants. CREPE's `CONFIDENCE` and `MODEL_SIZE` are deliberately
    # not here: Melodia has no confidence threshold and no model size, and a contract that
    # demanded them would be a contract about CREPE.
    for const in ("SR", "HOP", "HOP_SECONDS", "TRACKER"):
        assert hasattr(pitch, const), const


def test_track_takes_no_tonic():
    """A tonic in this signature would make the track un-shareable — only the folding
    downstream needs to know where Sa is."""
    import inspect
    params = inspect.signature(pitch.track).parameters
    assert not [p for p in params if "tonic" in p.lower()]


def test_track_finds_a_known_pitch():
    t = pitch.track(tone())
    voiced = np.asarray(t.voiced, dtype=bool)
    assert voiced.mean() > 0.5, f"only {voiced.mean():.0%} of a steady tone came back voiced"
    f0 = np.asarray(t.f0_hz, dtype=float)[voiced]
    cents = float(np.median(1200 * np.log2(f0 / TONE_HZ)))
    assert abs(cents) < CENTS_TOLERANCE, f"{TONE_HZ:g} Hz tracked {cents:+.0f} cents off"


def test_track_carries_what_callers_need():
    t = pitch.track(tone(seconds=2.0))
    assert np.shape(t.f0_hz) == np.shape(t.voiced)
    assert t.hop_seconds == pitch.HOP / pitch.SR
    assert 1.5 < t.seconds < 2.5
    assert 0.0 <= t.voiced_fraction <= 1.0
    cents = t.cents_above(TONIC)
    assert cents.ndim == 1 and np.isfinite(cents).all()


def test_cents_above_is_not_folded_into_one_octave():
    """Register is information — ambit and saptak distribution are both questions about
    octaves, and folding here would throw that away. A note a twelfth above Sa has to read
    as ~1900 cents, not ~700."""
    t = pitch.track(tone(hz=TONIC * 3, seconds=2.0))
    cents = t.cents_above(TONIC)
    assert cents.size, "nothing voiced"
    assert np.median(cents) > 1200.0, f"median {np.median(cents):.0f} cents — folded"


def test_the_same_recording_gives_the_same_track():
    """Melodia is deterministic, so this passes for free — and that is worth a test anyway.

    CREPE was not: it dithered its bins from numpy's *global* RNG, and unseeded the model
    answered differently every process — measured once at top-1 between 0.39 and 0.61 on
    one clip. This check is what would catch a tracker that reintroduced that.
    """
    a, b = pitch.track(tone()), pitch.track(tone())
    assert np.array_equal(np.asarray(a.f0_hz), np.asarray(b.f0_hz))
    assert np.array_equal(np.asarray(a.voiced), np.asarray(b.voiced))


def test_tracking_does_not_disturb_the_callers_rng():
    """Quietly reseeding numpy is not a reasonable side effect of asking for a pitch track."""
    np.random.seed(1234)
    before = np.random.get_state()
    pitch.track(tone(seconds=1.0))
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# ---------------------------------------------------------------- what reads the track

def test_histogram_is_a_distribution_over_time():
    f0, voiced = synthetic_track()
    h = np.asarray(melody_branch.histogram(f0, voiced, TONIC), dtype=float)
    assert h.shape == (melody_branch.N_BINS,)
    assert abs(h.sum() - 1.0) < 1e-9
    assert (h >= 0).all()


def test_a_swar_held_twice_as_long_is_twice_as_tall():
    """The property that makes the histogram drawable as "share of your time". A compressed
    array fails this — the power 0.5 would make it 1.41x."""
    f0 = np.concatenate([np.full(2000, TONIC), np.full(4000, TONIC * 2 ** (7 / 12))])
    h = np.asarray(melody_branch.histogram(f0, np.ones(f0.size, bool), TONIC), dtype=float)
    assert 1.8 < h[70] / h[0] < 2.2, f"ratio {h[70] / h[0]:.2f}"


def test_features_is_the_compression_and_is_kept_separate():
    f0, voiced = synthetic_track(seed=1)
    h = melody_branch.histogram(f0, voiced, TONIC)
    f = np.asarray(melody_branch.features(h), dtype=float)
    assert f.shape == np.shape(h)
    assert abs(f.sum() - 1.0) < 1e-9
    # compression pulls the tall bins down relative to the short ones
    assert f.max() < np.max(h) + 1e-12


def test_features_of_a_normalised_histogram_is_scale_free():
    """Normalising before the power and after it give the same answer, which is what lets
    the histogram be a distribution and the classifier's input be derived from it."""
    f0, voiced = synthetic_track(seed=2)
    h = np.asarray(melody_branch.histogram(f0, voiced, TONIC), dtype=float)
    direct = h ** melody_branch.POWER
    direct = direct / direct.sum()
    assert np.allclose(melody_branch.features(h), direct, rtol=0, atol=1e-15)


def test_a_histogram_without_building_a_classifier():
    """The whole point of the split, in four lines and no weights."""
    track = pitch.from_audio(tone(seconds=2.0), pitch.SR)
    h = np.asarray(melody_branch.profile(track, TONIC), dtype=float)
    assert abs(h.sum() - 1.0) < 1e-9
    direct = melody_branch.histogram(track.f0_hz, track.voiced, TONIC)
    assert np.array_equal(h, np.asarray(direct, dtype=float))


def test_from_audio_resamples():
    """Callers should not have to know the tracker's rate.

    The rate here is deliberately *not* `pitch.SR` — at 44.1 kHz this would pass whether
    `from_audio` resampled or not, and the point is that it does.
    """
    assert OTHER_SR != pitch.SR, "pick a rate unlike the tracker's, or this proves nothing"
    track = pitch.from_audio(tone(seconds=1.5, sr=OTHER_SR), OTHER_SR)
    assert 1.2 < track.seconds < 1.8
    voiced = np.asarray(track.voiced, dtype=bool)
    assert voiced.mean() > 0.5, f"only {voiced.mean():.0%} voiced after resampling"
    f0 = np.asarray(track.f0_hz, dtype=float)[voiced]
    cents = float(np.median(1200 * np.log2(f0 / TONE_HZ)))
    assert abs(cents) < CENTS_TOLERANCE, (
        f"{TONE_HZ:g} Hz at {OTHER_SR} Hz tracked {cents:+.0f} cents off — "
        f"a rate mismatch shows up here first")


def test_the_tracker_is_named():
    """Anything caching a track keys it by this, so a CREPE histogram and a Melodia one
    cannot be read back as each other."""
    assert isinstance(pitch.TRACKER, str) and pitch.TRACKER


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok    {name[5:].replace('_', ' ')}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name[5:].replace('_', ' ')}\n        {exc}")
    print(f"\n{'all checks passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
