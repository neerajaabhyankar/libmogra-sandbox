import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

import librosa


def get_audio(audio_file) -> (List[float], int):
    return librosa.load(audio_file)


def get_spectrogram(audio_array, verbose=True):
    D = librosa.stft(audio_array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    if verbose:
        plt.figure().set_figwidth(12)
        librosa.display.specshow(S_db, x_axis="time", y_axis="hz", bins_per_octave=22)
        plt.colorbar()
    return S_db
