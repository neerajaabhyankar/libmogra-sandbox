from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    # Set in each subclass; used as the output subdirectory name.
    name: str

    # If True, embed.py skips chunk_audio and passes the whole clip as a single
    # "chunk". For models with no meaningful sub-clip temporal structure (e.g.
    # global-avg-pooled classifiers with a fixed/minimum input length).
    whole_clip: bool = False

    @abstractmethod
    def load(self):
        """Download / load weights into memory. Called once before embedding."""
        ...

    @abstractmethod
    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """Embed a single audio chunk. Returns a 1-D array of shape (d,)."""
        ...
