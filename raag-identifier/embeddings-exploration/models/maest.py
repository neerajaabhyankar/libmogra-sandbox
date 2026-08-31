# MAEST: Music Audio Efficient Spectrogram Transformer
# ViT-style model trained on Discogs metadata; outputs 2304-D vectors from
# CLS + DIST + averaged patch tokens. Native checkpoints for 5/10/20/30s clips.
# Ref: https://github.com/palonso/MAEST   HF: MTG/discogs-maest-*
#
# TODO: implement once MTG/discogs-maest-30s-pw is confirmed on HuggingFace.
# The model uses a custom pipeline; check the repo for the inference snippet.

from .base import BaseEmbedder


class MAESTEmbedder(BaseEmbedder):
    name = "maest"

    def load(self):
        raise NotImplementedError("MAEST not yet implemented — see TODO in this file.")

    def embed(self, audio_array, sr):
        raise NotImplementedError
