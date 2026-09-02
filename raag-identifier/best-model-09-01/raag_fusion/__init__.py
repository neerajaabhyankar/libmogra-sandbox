"""Hindustani raag identification from a hummed or sung recording, given the tonic.

    from raag_fusion import RaagIdentifier
    model = RaagIdentifier.load()
    print(model.predict_file("alap.wav", tonic_hz=146.8))

See README.md for what it is, how well it works, and what it cannot do.
"""

from .identifier import Prediction, RaagIdentifier   # noqa: F401
from .tonic import from_hum as tonic_from_hum        # noqa: F401

__all__ = ["RaagIdentifier", "Prediction", "tonic_from_hum"]
__version__ = "1.0.0"
