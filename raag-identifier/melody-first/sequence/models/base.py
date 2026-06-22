from abc import ABC, abstractmethod
import numpy as np


class BaseSeqEmbedder(ABC):
    name: str

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """Return (T, D) float32 array — the per-frame/per-segment sequence embedding."""
        ...
