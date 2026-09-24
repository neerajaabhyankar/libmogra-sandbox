"""The tool: given audio (or a pitch track) and a swar samooha, find where it occurs.

    from pakad import find
    for occ in find("raga.mp3", ",n S m", tonic_hz=155.06):
        print(occ.t0, occ.t1, occ.probability)

    poetry run python pakad.py raga.mp3 --samooha ",n S m" --tonic 155.06 --top 5

Deliberately small surface: everything a caller needs is the samooha as text, the tonic, and a
threshold. The tonic is required and never guessed -- it is the one input that changes every
answer (see plan.md).
"""

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np

import _bootstrap  # noqa: F401
import calibrate
import config as C
import contour as contour_mod
import matcher
from utils import raagdb


@dataclass
class Occurrence:
    t0: float
    t1: float
    cost: float               # lower is better; not comparable across samoohas
    probability: float        # calibrated P(a listener calls this the samooha); coarse, see below
    swars: str                # the samooha, as asked for
    detail: dict              # pitch misfit, ornament fraction, wrong steps

    def __repr__(self):
        return f"<{self.swars} {self.t0:.2f}-{self.t1:.2f}s p={self.probability:.2f}>"


def parse_samooha(text):
    """',n S m' -> ((10, 0, 5), (-1, 0, 0)). Saptak marks are optional but respected."""
    swars, octaves = raagdb.parse_phrase(text.split())
    if len(swars) != len(text.split()):
        raise ValueError(f"could not parse samooha {text!r}")
    if len(swars) < 2:
        raise ValueError("a samooha needs at least two swars")
    return tuple(swars), tuple(octaves)


def contour_of(source, tonic_hz=None, sr=None):
    """A Contour from: a Contour, an (f0_hz, hop) pair, or a path to audio."""
    if hasattr(source, "cents"):
        return source
    if isinstance(source, tuple):
        f0, hop = source
        return contour_mod.contour_from_f0(np.asarray(f0), hop, tonic_hz, "given")
    import librosa
    import fullaudio
    audio, file_sr = librosa.load(source, sr=sr, mono=True)
    f0, _conf, hop = fullaudio._melodia(audio, file_sr)
    return contour_mod.contour_from_f0(f0, hop, tonic_hz, str(source))


def find(source, samooha, tonic_hz=None, top_k=C.TOP_K, min_probability=0.0, params=None):
    """Occurrences of `samooha`, best first.

    `source`: audio path, a `Contour`, or `(f0_hz, hop_seconds)`. `tonic_hz` is required unless
    a Contour is passed (it is already tonic-relative).
    """
    swars, octaves = parse_samooha(samooha)
    ctr = contour_of(source, tonic_hz)
    model = json.loads(calibrate.MODEL_JSON.read_text()) if calibrate.MODEL_JSON.exists() else None
    out = []
    for c in matcher.match(ctr, swars, top_k=top_k, params=params, octaves=octaves):
        p = float(calibrate.probability(c.cost, model)) if model else float("nan")
        if p >= min_probability or np.isnan(p):
            out.append(Occurrence(round(c.t0, 3), round(c.t1, 3), round(c.cost, 3), round(p, 3),
                                  samooha, dict(pitch_cost=round(c.pitch_cost, 3),
                                                ornament_fraction=round(c.orn_frac, 3),
                                                wrong_steps=c.leaps)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio")
    ap.add_argument("--samooha", required=True, help='e.g. ",n S m"')
    ap.add_argument("--tonic", type=float, required=True, help="Sa in Hz")
    ap.add_argument("--top", type=int, default=C.TOP_K)
    ap.add_argument("--min-probability", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    occ = find(a.audio, a.samooha, tonic_hz=a.tonic, top_k=a.top, min_probability=a.min_probability)
    if a.json:
        print(json.dumps([asdict(o) for o in occ], indent=1))
    else:
        for o in occ:
            print(f"{o.t0:8.2f} - {o.t1:6.2f} s   p={o.probability:.2f}   cost={o.cost:.3f}   "
                  f"{o.detail}")
