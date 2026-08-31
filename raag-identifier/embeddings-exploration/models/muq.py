# MuQ: large music foundation model (~300M params), current SOTA on many MIR tasks.
# Weights are CC-BY-NC 4.0 (non-commercial only).
# Ref: https://huggingface.co/mulab-ai/MuQ
#
# TODO: MuQ uses a custom AutoModel; confirm the HF repo supports
# transformers AutoModel.from_pretrained before wiring up here.
# Expected usage is similar to MERT (Wav2Vec2-style feature extractor).

from .base import BaseEmbedder


class MuQEmbedder(BaseEmbedder):
    name = "muq"

    def load(self):
        raise NotImplementedError("MuQ not yet implemented — see TODO in this file.")

    def embed(self, audio_array, sr):
        raise NotImplementedError
