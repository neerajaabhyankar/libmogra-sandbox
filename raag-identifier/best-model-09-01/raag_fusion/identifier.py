"""The model: two branches, calibrated, averaged, top-5 out.

    from raag_fusion import RaagIdentifier
    model = RaagIdentifier.load()
    for p in model.predict_file("alap.wav", tonic_hz=146.8):
        print(p.raag, p.probability)

**Why the two are averaged rather than concatenated.** Both were tried. The CQT network and
the pitch histogram disagree on 71 % of test clips while each is right about 37 % of the
time, so between them they hold the right answer for 55 %. Averaging their probabilities
captures 73 % of that pool and lands at 0.440. Feeding the histogram into the network as an
extra input instead captures 54 % and lands at 0.364 -- the network leans on the easy
feature and stops training the trunk. The cheap combination won.

**Calibration.** The two branches produce scores on completely different scales, so each is
turned into a probability distribution by a softmax whose temperature was fitted on the
validation split. Only then are they mixed, with a weight (0.40 on the histogram) also
chosen on validation. Both numbers are frozen in `weights/config.json`.

**Long recordings** are cut into 20 s windows -- the length every training clip had -- and
the per-window probabilities are averaged.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import audio, cqt_branch, melody_branch

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


@dataclass(repr=False)
class Prediction:
    raag: str
    probability: float

    def __str__(self):
        return f"{self.raag:<20s} {self.probability:6.1%}"

    def __repr__(self):
        # the generated dataclass repr prints float32 noise (0.4300000071525574), which
        # makes a list of five of these unreadable in a notebook
        return f"Prediction(raag={self.raag!r}, probability={self.probability:.3f})"


def _softmax(x, temperature=1.0):
    z = np.asarray(x, dtype=np.float64) / max(temperature, 1e-6)
    e = np.exp(z - z.max())
    return e / e.sum()


class RaagIdentifier:
    """Sa-anchored CQT network + pitch-histogram model, fused."""

    def __init__(self, net, linear, raags, config, device="cpu"):
        self.net = net.to(device).eval()
        self.linear = linear
        self.raags = list(raags)
        self.config = dict(config)
        self.device = device

    @classmethod
    def load(cls, weights_dir=WEIGHTS, device="cpu"):
        weights_dir = Path(weights_dir)
        config = json.loads((weights_dir / "config.json").read_text())
        raags = json.loads((weights_dir / "raags.json").read_text())
        net = cqt_branch.Net(n_raags=len(raags), lam=config["db_lam"])
        net.load_state_dict(torch.load(weights_dir / "cqt_net.pt", map_location=device,
                                       weights_only=True))
        linear = melody_branch.LinearModel.load(weights_dir / "melody_linear.npz")
        return cls(net, linear, raags, config, device=device)

    # ---------------------------------------------------------------- one 20 s window

    def _window_probabilities(self, y, sr, tonic_hz):
        """(p_fused, p_cqt, p_melody) for one window."""
        c = self.config

        self.net.eval()          # batch norm on a single window needs it, always
        x = cqt_branch.features(
            audio.fit_length(audio.peak_normalise(audio.resample(y, sr, audio.SR_CQT)),
                             int(round(audio.SR_CQT * audio.WINDOW_SECONDS))),
            tonic_hz)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(x)[None].float().to(self.device))
        p_cqt = _softmax(logits[0].cpu().numpy(), c["temperature_cqt"])

        f0, voiced = melody_branch.f0_track(audio.resample(y, sr, melody_branch.SR),
                                            device=self.device)
        hist = melody_branch.histogram(f0, voiced, tonic_hz)
        p_mel = _softmax(self.linear.scores(hist), c["temperature_melody"])

        w = c["melody_weight"]
        return (1.0 - w) * p_cqt + w * p_mel, p_cqt, p_mel

    # ---------------------------------------------------------------- public

    def probabilities(self, y, sr, tonic_hz):
        """(n_raags,) fused probabilities, averaged over the recording's 20 s windows."""
        if not tonic_hz or not np.isfinite(tonic_hz):
            raise ValueError("tonic_hz is required -- see raag_fusion.tonic.from_hum")
        ws = audio.windows(np.asarray(y, dtype=np.float32), sr)
        return np.mean([self._window_probabilities(w, sr, tonic_hz)[0] for w in ws], axis=0)

    def predict(self, y, sr, tonic_hz, top_k=5):
        p = self.probabilities(y, sr, tonic_hz)
        return [Prediction(self.raags[i], float(p[i]))
                for i in np.argsort(-p)[:top_k]]

    def predict_file(self, path, tonic_hz, top_k=5):
        y, sr = audio.load(path)
        return self.predict(y, sr, tonic_hz, top_k=top_k)
