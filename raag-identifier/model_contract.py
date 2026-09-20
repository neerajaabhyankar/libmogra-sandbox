#!/usr/bin/env python3
"""Does this model version keep the pitch pathway convention?

    python3 model_contract.py best-model-09-01
    python3 model_contract.py some-new-model --package my_fusion
    python3 model_contract.py best-model-09-01 --quick     # skip anything that runs the tracker

Run it against a model directory before calling that version done. See CLAUDE.md in this
folder for what the convention is and why it exists; this file is the executable half.

It imports the model package by path and pokes at it, so it works on a directory that was
never installed. It does **not** need the dataset, the weights, or a network — the audio it
tests with is synthesised here.

Deliberately not pytest: a model directory is uploadable to the Hub and its own `tests/`
must stand alone, so the shared checker lives out here and each version keeps a small
self-contained test of its own.
"""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np

#: A rate deliberately unlike any tracker's, used only to prove `from_audio` resamples.
#: The tone every other check uses is synthesised at the package's own `pitch.SR`, because
#: `pitch.track` takes audio at that rate and nothing else -- 16 kHz samples handed to a
#: 44.1 kHz tracker read as a tone 2.76x too high, which is a bug in the checker and not
#: in the model.
OTHER_SR = 16000
TONIC = 146.83
TONE_HZ = 220.0
TONE_SECONDS = 3.0
CENTS_TOLERANCE = 35.0      # generous: trackers quantise, and some dither on top of it


def load_package(directory, name):
    path = Path(directory).resolve() / name
    if not (path / "__init__.py").exists():
        raise SystemExit(f"no package at {path}")
    sys.path.insert(0, str(Path(directory).resolve()))
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tone(sr, hz=TONE_HZ, seconds=TONE_SECONDS):
    """A steady note with a few partials — something any pitch tracker will follow."""
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * hz * t
    y = sum(a * np.sin(k * phase) for k, a in enumerate((1.0, 0.5, 0.25, 0.12), 1))
    return (y / np.max(np.abs(y))).astype(np.float32)


class Checks:
    """Each `check_*` returns None to pass, or a string saying what is wrong."""

    def __init__(self, pkg, name, quick):
        self.pkg, self.name, self.quick = pkg, name, quick
        self._track = None

    # -- structure, no CREPE needed -------------------------------------------------

    def check_pitch_module_exists(self):
        if not hasattr(self.pkg, "pitch"):
            return (f"{self.name}.pitch is missing — the tracker has to be its own module, "
                    f"not buried in the melody branch")
        return None

    def check_pitch_api(self):
        p = self.pkg.pitch
        for fn in ("track", "from_audio"):
            if not callable(getattr(p, fn, None)):
                return f"{self.name}.pitch.{fn} is missing"
        if not hasattr(p, "PitchTrack"):
            return f"{self.name}.pitch.PitchTrack is missing"
        # Tracker-neutral: the rate `track` expects, the frame rate, and which tracker
        # produced it. `CONFIDENCE`/`MODEL_SIZE` were CREPE's and are not required of a
        # tracker that has no confidence threshold.
        for const in ("SR", "HOP", "HOP_SECONDS", "TRACKER"):
            if not hasattr(p, const):
                return f"{self.name}.pitch.{const} is missing"
        return None

    def check_track_is_tonic_free(self):
        """The signature is the contract: a tonic here would make the track un-shareable."""
        params = inspect.signature(self.pkg.pitch.track).parameters
        bad = [n for n in params if "tonic" in n.lower()]
        return f"pitch.track takes {bad} — the track must be tonic-free" if bad else None

    def check_melody_api(self):
        m = getattr(self.pkg, "melody_branch", None)
        if m is None:
            return f"{self.name}.melody_branch is missing"
        for fn in ("histogram", "features", "profile"):
            if not callable(getattr(m, fn, None)):
                return (f"{self.name}.melody_branch.{fn} is missing — the histogram and the "
                        f"classifier's compression of it have to be separable")
        return None

    def check_histogram_is_a_distribution(self):
        """Synthetic frames, no CREPE: does histogram() sum to 1 and read as time?"""
        m = self.pkg.melody_branch
        rng = np.random.default_rng(0)
        f0 = TONIC * 2 ** rng.normal(0, 0.5, 4000)
        voiced = rng.random(4000) < 0.8
        h = np.asarray(m.histogram(f0, voiced, TONIC), dtype=float)
        if abs(h.sum() - 1.0) > 1e-9:
            return f"histogram sums to {h.sum():.6f}, not 1 — it is not a distribution"
        if (h < 0).any():
            return "histogram has negative bins"

        # doubling how long one pitch class is held must double its share, which is exactly
        # what a compressed feature vector would fail
        one = np.full(2000, TONIC)
        two = np.full(4000, TONIC * 2 ** (7 / 12))
        f0b = np.concatenate([one, two])
        vb = np.ones(f0b.size, dtype=bool)
        hb = np.asarray(m.histogram(f0b, vb, TONIC), dtype=float)
        lo, hi = hb[0], hb[70]
        if not (1.8 < hi / max(lo, 1e-12) < 2.2):
            return (f"a pitch class held twice as long is {hi / max(lo, 1e-12):.2f}x as "
                    f"tall, not ~2x — histogram() looks compressed; move that into features()")
        return None

    def check_features_is_separate_and_normalised(self):
        m = self.pkg.melody_branch
        rng = np.random.default_rng(1)
        f0 = TONIC * 2 ** rng.normal(0, 0.5, 3000)
        voiced = rng.random(3000) < 0.9
        h = m.histogram(f0, voiced, TONIC)
        f = np.asarray(m.features(h), dtype=float)
        if abs(f.sum() - 1.0) > 1e-9:
            return f"features() sums to {f.sum():.6f}, not 1"
        if f.shape != np.shape(h):
            return "features() changed the shape of the histogram"
        return None

    # -- behaviour, runs the tracker ------------------------------------------------

    def tone(self, seconds=TONE_SECONDS, hz=TONE_HZ):
        """A test tone at the package's own rate, which is what `pitch.track` takes."""
        return tone(self.pkg.pitch.SR, hz=hz, seconds=seconds)

    def track_once(self):
        if self._track is None:
            self._track = self.pkg.pitch.track(self.tone())
        return self._track

    def check_track_finds_a_known_pitch(self):
        t = self.track_once()
        voiced = np.asarray(t.voiced, dtype=bool)
        if voiced.sum() < 0.5 * voiced.size:
            return f"only {voiced.mean():.0%} of a steady {TONE_HZ:g} Hz tone came back voiced"
        f0 = np.asarray(t.f0_hz, dtype=float)[voiced]
        cents = float(np.median(1200 * np.log2(f0 / TONE_HZ)))
        if abs(cents) > CENTS_TOLERANCE:
            return f"a {TONE_HZ:g} Hz tone tracked {cents:+.0f} cents off"
        return None

    def check_track_fields(self):
        t = self.track_once()
        for field in ("f0_hz", "voiced", "hop_seconds"):
            if not hasattr(t, field):
                return f"PitchTrack has no {field}"
        for prop in ("seconds", "voiced_fraction"):
            if not hasattr(t, prop):
                return f"PitchTrack has no {prop}"
        if not callable(getattr(t, "cents_above", None)):
            return "PitchTrack.cents_above(tonic_hz) is missing"
        if np.shape(t.f0_hz) != np.shape(t.voiced):
            return "f0_hz and voiced are different lengths"
        if not 0.0 <= t.voiced_fraction <= 1.0:
            return f"voiced_fraction is {t.voiced_fraction}"
        return None

    def check_track_is_reproducible(self):
        """Same audio, same answer, in this process and the next.

        Melodia is deterministic and passes this for free. CREPE did not: it decoded to a
        20-cent grid and added dither from numpy's *global* RNG, so the check exists to
        catch a tracker that has gone back to drawing randomness it does not seed.
        """
        a, b = self.pkg.pitch.track(self.tone()), self.pkg.pitch.track(self.tone())
        if not np.array_equal(np.asarray(a.f0_hz), np.asarray(b.f0_hz)):
            return ("two tracks of the same audio differ — the tracker is drawing from an "
                    "unseeded RNG")
        return None

    def check_track_leaves_the_global_rng_alone(self):
        np.random.seed(1234)
        before = np.random.get_state()
        self.pkg.pitch.track(self.tone())
        after = np.random.get_state()
        same = before[0] == after[0] and np.array_equal(before[1], after[1]) \
            and before[2:] == after[2:]
        return None if same else "pitch.track reseeded numpy's global RNG and left it that way"

    def check_standalone_use(self):
        """The whole point: a histogram without constructing a classifier."""
        t = self.pkg.pitch.from_audio(tone(OTHER_SR, seconds=2.0), OTHER_SR)
        h = np.asarray(self.pkg.melody_branch.profile(t, TONIC), dtype=float)
        if abs(h.sum() - 1.0) > 1e-9:
            return "profile(track, tonic) is not a distribution"
        direct = np.asarray(
            self.pkg.melody_branch.histogram(t.f0_hz, t.voiced, TONIC), dtype=float)
        if not np.allclose(h, direct, rtol=0, atol=0):
            return "profile(track, tonic) disagrees with histogram(f0, voiced, tonic)"
        return None


QUICK_ONLY = {"check_pitch_module_exists", "check_pitch_api", "check_track_is_tonic_free",
              "check_melody_api", "check_histogram_is_a_distribution",
              "check_features_is_separate_and_normalised"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="a model version directory")
    ap.add_argument("--package", default="raag_fusion", help="package name inside it")
    ap.add_argument("--quick", action="store_true", help="skip everything that runs CREPE")
    a = ap.parse_args()

    pkg = load_package(a.directory, a.package)
    checks = Checks(pkg, a.package, a.quick)
    names = [n for n in dir(checks) if n.startswith("check_")]
    if a.quick:
        names = [n for n in names if n in QUICK_ONLY]

    failures = 0
    print(f"{a.package} in {a.directory}\n")
    for name in sorted(names):
        label = name[len("check_"):].replace("_", " ")
        try:
            problem = getattr(checks, name)()
        except Exception as exc:                     # a check that explodes is a failure
            problem = f"{type(exc).__name__}: {exc}"
        if problem:
            failures += 1
            print(f"  FAIL  {label}\n        {problem}")
        else:
            print(f"  ok    {label}")

    print(f"\n{len(names) - failures}/{len(names)} checks passed")
    if failures:
        print("this version does not keep the pitch pathway convention -- see CLAUDE.md")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
