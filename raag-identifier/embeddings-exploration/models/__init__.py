from .clap import CLAPEmbedder
from .mert import MERTEmbedder
from .maest import MAESTEmbedder
from .muq import MuQEmbedder
from .musicfm import MusicFMEmbedder
from .melodysim import MelodySimEmbedder
from .crc_jeevster import CRCJeevsterEmbedder

# Maps config.MODELS_TO_RUN names → embedder instances.
# Lambda wrappers let us pass constructor args without instantiating at import time.
REGISTRY = {
    "clap":         CLAPEmbedder,
    "mert-95m":     lambda: MERTEmbedder("m-a-p/MERT-v1-95M"),
    "mert-330m":    lambda: MERTEmbedder("m-a-p/MERT-v1-330M"),
    "maest":        MAESTEmbedder,
    "muq":          MuQEmbedder,
    "musicfm":      MusicFMEmbedder,
    "melodysim":    MelodySimEmbedder,
    "crc-jeevster": CRCJeevsterEmbedder,
}


def get_embedder(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]()
