import librosa
import numpy as np
from sklearn.neighbors import KernelDensity

FRAME_LENGTH = 2048
KDE_GRANULARITY = 240  # number of notes in an octave
KDE_GBANDWIDTH = 0.1


def extract(y, sr, tonic):

    # pitch tracking
    f0, _, _ = librosa.pyin(
        y, sr=sr,
        fmin=tonic/2, fmax=tonic*4,
        FRAME_LENGTH=FRAME_LENGTH
    )

    # convert frequency to MIDI for easier binning
    # needed since midi is in log scale already
    f0_midi = librosa.hz_to_midi(f0)

    # subtract tonic + fold into octave
    f0_relative = (f0_midi-librosa.hz_to_midi(tonic)) % 12
    f0_relative = f0_relative[~np.isnan(f0_relative)]

    # histogram + smooth with a kde
    supp = np.linspace(0, 12, KDE_GRANULARITY).reshape(-1, 1)
    np.random.seed(0)
    kde = KernelDensity(kernel="gaussian", bandwidth=KDE_GBANDWIDTH).fit(f0_relative.reshape(-1, 1))
    logkde = kde.score_samples(supp)
    kde_sample = np.exp(logkde)
    
    return kde_sample
