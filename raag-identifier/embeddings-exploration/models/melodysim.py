# MelodySim: MERT-v1-95M backbone fine-tuned with triplet loss for melody similarity.
# HuggingFace: amaai-lab/MelodySim  (checkpoint: siamese_net_20250328.ckpt)
#
# Architecture (reconstructed from the repo source):
#   1. MERT hidden states at layers [2, 5, 8, 11] (every 3rd, starting at 2)
#   2. AvgPool1d(kernel=10, stride=10) to reduce time dimension
#   3. Concatenate 4 × 768-dim → (batch, 3072, T_reduced)
#   4. SiameseNet: ResidualBlock(3072→512) → ResidualBlock(512→256) →
#      AdaptiveAvgPool1d(1) → Linear(256, 128) → (batch, 128)
#
# We inline the architecture so the MelodySim repo doesn't need to be cloned.

import numpy as np
import librosa
import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from .base import BaseEmbedder

TARGET_SR = 24000     # MERT's expected sample rate
EMBEDDING_DIM = 128   # siamese_emb_dim from the published config


# ── SiameseNet (inlined from amaai-lab/MelodySim) ────────────────────────────

class _ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride, pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride, pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.shortcut = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch)) \
            if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class _SiameseNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.layer1      = _ResidualBlock(3072, 512)
        self.layer2      = _ResidualBlock(512, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc          = nn.Linear(256, embedding_dim)

    def forward(self, x):  # x: (batch, 3072, T)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)


# ── Embedder ──────────────────────────────────────────────────────────────────

class MelodySimEmbedder(BaseEmbedder):
    name = "melodysim"

    def load(self):
        from huggingface_hub import hf_hub_download

        # Download checkpoint (~30 MB) to HF cache; skipped if already present.
        ckpt_path = hf_hub_download(
            repo_id="amaai-lab/MelodySim",
            filename="siamese_net_20250328.ckpt",
        )

        # Load just the siamese_net weights from the Lightning checkpoint.
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        siamese_state = {
            k.replace("siamese_net.", ""): v
            for k, v in state.items()
            if k.startswith("siamese_net.")
        }
        self.siamese_net = _SiameseNet()
        self.siamese_net.load_state_dict(siamese_state)
        self.siamese_net.eval()

        # MERT backbone (same as used during MelodySim training).
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        self.mert.eval()

        self._time_reduce = nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False)

    def _mert_features(self, audio_array: np.ndarray) -> torch.Tensor:
        """Run MERT, pick layers [2,5,8,11], time-reduce, concat → (1, 3072, T)."""
        inputs = self.processor(audio_array, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            hidden = self.mert(**inputs, output_hidden_states=True).hidden_states
        # hidden_states[2::3] → indices 2, 5, 8, 11 (4 layers × 768 dim)
        selected = hidden[2::3]   # tuple of 4 tensors, each (1, T, 768)
        reduced  = [
            self._time_reduce(h.squeeze(0).T).T   # (T, 768) → reduce time → back
            for h in selected
        ]
        # Stack → (4, T_reduced, 768), concat channels → (1, 3072, T_reduced)
        concat = torch.cat([r.T for r in reduced], dim=0).unsqueeze(0)  # (1, 3072, T_reduced)
        return concat

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        if sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
        features = self._mert_features(audio_array)          # (1, 3072, T)
        with torch.no_grad():
            emb = self.siamese_net(features).squeeze(0)      # (128,)
        return emb.numpy()
