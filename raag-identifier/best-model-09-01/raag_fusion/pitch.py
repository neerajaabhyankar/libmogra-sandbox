"""CREPE's pitch track, on its own.

Everything that reads melody out of a recording starts here — the raag classifier's
histogram, a swar histogram drawn for a listener, and whatever insights are built on top of
one. Splitting it out is what lets those share a single pass over the audio instead of each
running CREPE again.

**The track is tonic-free.** It is absolute Hz, so it can be computed before anyone has
decided where Sa is, and the same track answers every question that comes after. Only the
folding into swars needs a tonic, and that lives in `melody_branch`.

The numbers here are the ones the shipped model was measured with: 10 ms hop, tiny model,
periodicity 0.4 as the voiced threshold. Nothing in this module chooses them per call so
that two callers cannot silently disagree about what a pitch track is.
"""

from dataclasses import dataclass

import numpy as np

SR = 16000
HOP = 160                 # 10 ms
HOP_SECONDS = HOP / SR
CONFIDENCE = 0.4          # torchcrepe periodicity below this is treated as unvoiced
MODEL_SIZE = "tiny"
FMIN, FMAX = 50.0, 2000.0
BATCH_SIZE = 512
DITHER_SEED = 0


@dataclass(frozen=True)
class PitchTrack:
    """One f0 estimate per [HOP_SECONDS], and whether each was voiced.

    `f0_hz` is defined at every frame; `voiced` says which of them to believe. Unvoiced
    frames keep whatever CREPE guessed rather than being blanked, because callers that want
    a continuous contour for plotting need it and callers that want statistics mask it.
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


def track(y16000, device="cpu", dither_seed=DITHER_SEED):
    """[SR] mono audio -> a [PitchTrack].

    **The dither seed is not optional decoration.** torchcrepe decodes pitch to a 20-cent
    bin grid and then adds triangular noise of +-20 cents to every frame to hide the
    quantisation (`torchcrepe.convert.dither`), drawn from numpy's *global* RNG. Left
    alone, that makes this function return a different answer every process: measured on
    one 24 s clip, the model's top-1 probability moved between 0.39 and 0.61 and the
    ranking below first place reshuffled. Seeding fixes the draw, so the same recording
    always gets the same answer.

    The caller's RNG state is saved and restored, because quietly reseeding numpy is not
    a reasonable side effect of asking for a pitch track.
    """
    import torch
    import torchcrepe

    wav = torch.from_numpy(np.ascontiguousarray(y16000)).float().unsqueeze(0)
    state = np.random.get_state()
    try:
        np.random.seed(dither_seed)
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                wav, SR, hop_length=HOP, fmin=FMIN, fmax=FMAX, model=MODEL_SIZE,
                return_periodicity=True, batch_size=BATCH_SIZE, device=device,
                decoder=torchcrepe.decode.weighted_argmax)
    finally:
        np.random.set_state(state)
    return PitchTrack(f0.squeeze(0).numpy(), periodicity.squeeze(0).numpy() >= CONFIDENCE)


def from_audio(y, sr, device="cpu", dither_seed=DITHER_SEED):
    """Audio at any rate -> a [PitchTrack], resampling to [SR] first."""
    from . import audio

    return track(audio.resample(y, sr, SR), device=device, dither_seed=dither_seed)
