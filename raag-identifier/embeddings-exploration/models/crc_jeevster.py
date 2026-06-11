# CRC-Jeevster: pretrained Carnatic raga classifier (1D-ResNet over raw audio).
# Source: carnatic-raga-classifier-jeevster (symlinked, read-only). See crc_jeevster.md
# for the full architecture writeup and rationale for the chosen embedding layer.
#
# Architecture: Conv1d stem (2->300ch, k=80, stride=16) -> 10x ResidualBlock(300ch,
# k=3) each followed by MaxPool1d(2) -> global avg-pool over time -> Linear(300->150)
# -> log_softmax. Trained on 30s @ 8kHz stereo clips, per-channel normalized.
#
# We use the 300-dim global-avg-pooled vector (fc1's input) as the embedding -- the
# standard penultimate-layer embedding for a classifier.
#
# whole_clip = True: the model has a hard minimum input length (~2-5s @ 8kHz; below
# that max_pool1d collapses to size 0) and no meaningful sub-clip temporal structure
# (avg-pool is global), so we embed each clip as a single unit rather than chunking.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import librosa
import torch

from .base import BaseEmbedder

JEEVSTER_DIR = Path(__file__).resolve().parent.parent.parent / "carnatic-raga-classifier-jeevster"
TARGET_SR = 8000
MIN_SAMPLES = 40000  # 5s @ 8kHz -- shorter inputs crash in max_pool1d (see crc_jeevster.md)


class CRCJeevsterEmbedder(BaseEmbedder):
    name = "crc-jeevster"
    whole_clip = True

    def load(self):
        # Load jeevster's models.py under a distinct module name -- "models" is
        # already taken by this package (embeddings-exploration/models/__init__.py).
        spec = importlib.util.spec_from_file_location("jeevster_models", JEEVSTER_DIR / "models.py")
        jeevster_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jeevster_models)
        ResNetRagaClassifier = jeevster_models.ResNetRagaClassifier

        params = SimpleNamespace(
            input_channels=2, n_channel=300, stride=16,
            n_blocks=10, max_pool_every=1, num_classes=150,
        )
        self.model = ResNetRagaClassifier(params)
        ckpt = torch.load(JEEVSTER_DIR / "ckpts" / "best_ckpt.tar", map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self._captured = {}
        self.model.fc1.register_forward_hook(
            lambda module, inp, out: self._captured.__setitem__("emb", inp[0].detach())
        )

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        if sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)

        x = torch.from_numpy(audio_array).float().unsqueeze(0).repeat(2, 1)  # mono -> stereo (2, T)

        if x.shape[1] < MIN_SAMPLES:
            x = torch.nn.functional.pad(x, (0, MIN_SAMPLES - x.shape[1]))

        # per-channel normalize over time, matching dataloader.py
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

        with torch.no_grad():
            self.model(x.unsqueeze(0))

        return self._captured["emb"].squeeze().numpy()  # (300,)
