"""Every constant of the transcriber package."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"                       # cache/<model>/<key>.npz -- not committed
RAAG_IDENTIFIER = (HERE / ".." / "raag-identifier").resolve()
MELODY_EXTRACTION = RAAG_IDENTIFIER / "melody-extraction"
SOURCE_SEPARATION = RAAG_IDENTIFIER / "source-separation"

DEVICE = "mps"                                   # torch device for models that use torch

CREPE = dict(model="full", sr=16000, hop=160, voiced_conf=0.4)   # fmin 50, fmax 2000 Hz: set in crepe_tracker
