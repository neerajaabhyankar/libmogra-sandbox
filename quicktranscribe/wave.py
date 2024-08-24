import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

import soundfile as sf
import librosa


def get_audio(audio_file) -> Tuple[List[float], int]:
    return librosa.load(audio_file)


def read_audio_section(filename, start_time, stop_time):
    track = sf.SoundFile(filename)
    if not track.seekable():
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate
    start_frame = sr * start_time
    frames_to_read = sr * (stop_time - start_time)
    track.seek(start_frame)
    audio_section = track.read(frames_to_read)
    
    # sf.write(output_filename, audio_extract, sr)  # to write
    
    return audio_section, sr


def get_spectrogram(audio_array, verbose=True):
    D = librosa.stft(audio_array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    if verbose:
        plt.figure().set_figwidth(12)
        librosa.display.specshow(S_db, x_axis="time", y_axis="hz", bins_per_octave=22)
        plt.colorbar()
    return S_db
