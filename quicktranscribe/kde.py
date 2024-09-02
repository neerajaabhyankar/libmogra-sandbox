import librosa
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks

FRAME_LENGTH = 2048
KDE_GRANULARITY = 240  # number of notes in an octave
KDE_GBANDWIDTH = 0.1


def extract(y, sr, tonic):

    # pitch tracking
    f0, _, _ = librosa.pyin(
        y, sr=sr,
        fmin=tonic/2, fmax=tonic*4,
        frame_length=FRAME_LENGTH
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


def prominence_based_peak_finder(data, prominence=0.1, width=None, distance=None):
    extension_size = int(len(data) * 0.1)
    extended_data = np.concatenate([data[-extension_size:], data, data[:extension_size]])

    extended_peaks, properties = find_peaks(extended_data, prominence=prominence, width=width, distance=distance)
    peaks = []
    for peak in extended_peaks:
        if extension_size <= peak < extension_size + len(data):
            peaks.append(peak - extension_size)
    
    return peaks, properties


def derivative_based_peak_finder(data, threshold1=1e-2, threshold2=1e-3):
    extension_size = int(len(data) * 0.1)
    extended_data = np.concatenate([data[-extension_size:], data, data[:extension_size]])
    
    # Calculate first and second derivatives
    first_derivative = np.gradient(extended_data)
    second_derivative = np.gradient(first_derivative)
    
    extended_peaks = []
    for i in range(1, len(data) - 1):
        # Check if first derivative is near zero (flat) and second derivative is negative
        if abs(first_derivative[i]) < threshold1 and second_derivative[i] < -threshold2:
            extended_peaks.append(i)
    
    peaks = []
    for peak in extended_peaks:
        if extension_size <= peak < extension_size + len(data):
            peaks.append(peak - extension_size)
    
    return peaks