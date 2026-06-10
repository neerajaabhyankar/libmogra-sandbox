# MusicFM: music foundation model (~300M params).
# Note from the authors: self-supervised music models (including MusicFM) can
# have low key-detection performance without fine-tuning — worth keeping in mind
# when interpreting melody-related results.
# Ref: https://huggingface.co/minzwon/musicfm (0.3B checkpoint)
#
# TODO: the MusicFM repo has a custom inference API; confirm checkpoint format
# and wire up here following the same BaseEmbedder interface.

from .base import BaseEmbedder


class MusicFMEmbedder(BaseEmbedder):
    name = "musicfm"

    def load(self):
        raise NotImplementedError("MusicFM not yet implemented — see TODO in this file.")

    def embed(self, audio_array, sr):
        raise NotImplementedError
