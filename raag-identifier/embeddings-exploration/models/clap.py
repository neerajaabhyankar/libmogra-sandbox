# LAION-CLAP (music variant)
# Joint audio-text contrastive model; useful as a semantic baseline.
# Not melody-specific — will likely cluster by genre/mood/timbre first.
# Checkpoint: laion/larger_clap_music (music-specific, ~150M params, Mac-friendly)

import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor

from .base import BaseEmbedder

TARGET_SR = 48000  # CLAP requires 48 kHz input


class CLAPEmbedder(BaseEmbedder):
    name = "clap"

    def load(self):
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
        self.model = ClapModel.from_pretrained("laion/larger_clap_music")
        self.model.eval()

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        if sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
        inputs = self.processor(audios=audio_array, return_tensors="pt", sampling_rate=TARGET_SR)
        with torch.no_grad():
            features = self.model.get_audio_features(**inputs)
        return features.squeeze().numpy()
