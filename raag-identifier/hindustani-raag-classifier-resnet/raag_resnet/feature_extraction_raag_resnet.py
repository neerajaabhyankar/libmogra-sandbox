# Preprocessing for RaagResNetForAudioClassification, ported from
# embeddings-exploration/models/crc_jeevster.py's CRCJeevsterEmbedder.embed:
# resample to 8kHz, mono->stereo, zero-pad to a minimum length, per-channel
# normalize over time.

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import FeatureExtractionMixin


class RaagResNetFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_values"]

    def __init__(self, sampling_rate: int = 8000, min_input_samples: int = 40000, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.min_input_samples = min_input_samples

    def __call__(self, audio_array: np.ndarray, sampling_rate: int) -> dict:
        if sampling_rate != self.sampling_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=self.sampling_rate)

        x = torch.from_numpy(audio_array).float().unsqueeze(0).repeat(2, 1)  # mono -> stereo (2, T)

        if x.shape[1] < self.min_input_samples:
            x = F.pad(x, (0, self.min_input_samples - x.shape[1]))

        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

        return {"input_values": x}  # (2, T)
