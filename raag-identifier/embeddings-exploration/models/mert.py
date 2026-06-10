# MERT: Music Understanding Model with large-scale self-supervised training.
# Two checkpoints: 95M (fast, safe on 16 GB) and 330M (slower but stronger).
# Operates at 24 kHz. Outputs hidden states from 12/24 transformer layers.
# We use the last hidden state mean-pooled over time as the chunk embedding.
# Ref: https://huggingface.co/m-a-p/MERT-v1-95M

import numpy as np
import librosa
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from .base import BaseEmbedder

TARGET_SR = 24000


class MERTEmbedder(BaseEmbedder):

    def __init__(self, model_id: str):
        self.model_id = model_id
        # Derive a short name for the output directory, e.g. "mert-v1-95m"
        self.name = model_id.split("/")[-1].lower()

    def load(self):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self.model.eval()

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        if sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
        inputs = self.processor(audio_array, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # last_hidden_state: (1, T, d) → mean over T → (d,)
        return outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
