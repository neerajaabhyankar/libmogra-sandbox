"""Essentia's Melodia pitch track, on its own.

Everything that reads melody out of a recording starts here — the raag classifier's
histogram, a swar histogram drawn for a listener, and whatever insights are built on top of
one. Splitting it out is what lets those share a single pass over the audio instead of each
running the tracker again.

**The track is tonic-free.** It is absolute Hz, so it can be computed before anyone has
decided where Sa is, and the same track answers every question that comes after. Only the
folding into swars needs a tonic, and that lives in `melody_branch`.

The settings are the ones behind Saraga's and Dunya's published pitch annotations: 44.1 kHz,
`EqualLoudness` first, 2048-sample frames, a 196-sample hop (225 fps), 10-cent bins, and no
guessing through unvoiced stretches. They were reproduced against a Saraga `.pitch.txt` to a
median error of 0.00 cents. Nothing in this module chooses them per call, so two callers
cannot silently disagree about what a pitch track is.

**Why Melodia and not CREPE.** Melodia is a *predominant*-melody tracker built for
polyphonic audio: it forms pitch contours and filters them for voicing and octave errors
before choosing the melody. CREPE is a frame-wise neural estimator with no notion of a
contour. Three consequences, all measured on this corpus:

* It is about **14x faster** — 57x real time against CREPE-tiny's 4x, both on CPU — and it
  needs no GPU, so it does not contend with a training run for the accelerator.
* Its track is far smoother: 0.7 % of frame-to-frame steps exceed 100 cents, against
  CREPE's 5.3 %.
* It is **deterministic**. CREPE decoded pitch to a 20-cent grid and then added triangular
  dither drawn from numpy's *global* RNG, so the same recording gave a different answer
  every process unless the seed was pinned. Melodia has no such step, and the seeding
  machinery this module used to carry is gone.

It keeps fewer frames than CREPE did — 74 % of the corpus voiced against 80 % (medians) —
and that is the point rather than a cost: the frames it drops are the ones it could not fit
to a melodic contour.
"""

from dataclasses import dataclass

import numpy as np

#: Identifies the tracker. Also the key under which callers cache its output, so a track
#: made by one tracker can never be read back as another's.
TRACKER = "essentia-melodia"

SR = 44100                # Melodia's rate, and the rate `track` expects
HOP = 196                 # 225 frames per second
HOP_SECONDS = HOP / SR
FRAME_SIZE = 2048
BIN_RESOLUTION = 10       # cents per bin; the output lands on this grid
GUESS_UNVOICED = False    # do not invent pitch where no contour was found
EQUAL_LOUDNESS = True     # Melodia's salience function assumes it


@dataclass(frozen=True)
class PitchTrack:
    """One f0 estimate per [HOP_SECONDS], and whether each was voiced.

    `f0_hz` is defined at every frame; `voiced` says which of them to believe. Melodia
    reports an unvoiced frame as 0 Hz, which is kept rather than blanked, because callers
    that want a continuous contour for plotting need a value at every frame and callers
    that want statistics mask it.
    """

    f0_hz: np.ndarray
    voiced: np.ndarray
    hop_seconds: float = HOP_SECONDS

    def __len__(self):
        return int(self.f0_hz.size)

    @property
    def seconds(self):
        return len(self) * self.hop_seconds

    @property
    def voiced_fraction(self):
        """How much of the passage had a clear pitch at all — a recording that is 30 %
        voiced deserves less trust than one that is 80 %, and nothing else says so."""
        return float(np.mean(self.voiced)) if len(self) else 0.0

    def cents_above(self, tonic_hz):
        """Voiced frames as cents above [tonic_hz], unfolded, with non-finite dropped.

        Unfolded because register is information: the ambit of a performance and how its
        time splits between the saptaks are both questions about octaves, and folding
        throws that away.
        """
        f0 = np.asarray(self.f0_hz, dtype=float)[np.asarray(self.voiced, dtype=bool)]
        cents = 1200.0 * np.log2(np.clip(f0, 1e-6, None) / float(tonic_hz))
        return cents[np.isfinite(cents)]


def track(y, device=None):
    """[SR] mono audio -> a [PitchTrack].

    `device` is accepted and ignored. Melodia is a CPU algorithm, and the argument is kept
    so that callers written against the CREPE pathway keep working unchanged — passing
    `device="mps"` is not an error, it simply has nothing to select.

    The result is a pure function of the samples: the same audio gives the same track in
    every process, with no seed to pin and no global RNG touched.
    """
    import essentia
    import essentia.standard as es

    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).ravel())
    was_active = essentia.log.warningActive
    essentia.log.warningActive = False        # one warning per frame is not useful output
    try:
        signal = es.EqualLoudness(sampleRate=SR)(y) if EQUAL_LOUDNESS else y
        f0_hz, _confidence = es.PredominantPitchMelodia(
            sampleRate=SR, frameSize=FRAME_SIZE, hopSize=HOP,
            binResolution=BIN_RESOLUTION, guessUnvoiced=GUESS_UNVOICED)(signal)
    finally:
        essentia.log.warningActive = was_active

    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    return PitchTrack(f0_hz, f0_hz > 0)       # Melodia says "unvoiced" by reporting 0 Hz


def from_audio(y, sr, device=None):
    """Audio at any rate -> a [PitchTrack], resampling to [SR] first."""
    from . import audio

    return track(audio.resample(y, sr, SR), device=device)
