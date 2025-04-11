import librosa
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks

FRAME_LENGTH = 2048
KDE_GRANULARITY = 240  # number of notes in an octave
KDE_GBANDWIDTH = 0.1


def extract(y, sr, tonic):
    """
    given a waveform and sampling rate, outputs a histogram of KDE_GRANULARITY bins that span the octave
    that indicate the octave-folded f0 concentration at that relative frequency
    note: we need a tonic input relative to which we set the midi scale
    """

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
    # TODO(neeraja): loop & smooth across octave boundary
    kde = KernelDensity(kernel="gaussian", bandwidth=KDE_GBANDWIDTH).fit(f0_relative.reshape(-1, 1))
    logkde = kde.score_samples(supp)
    kde_sample = np.exp(logkde)
    
    # set area under curve to 1
    kde_sample = kde_sample / np.trapz(kde_sample, dx=12/KDE_GRANULARITY)
    
    return kde_sample


def frequency_from_dist_idx(kde_idx, tonic):
    f0_midi = librosa.hz_to_midi(tonic)
    midi_idx = kde_idx / (KDE_GRANULARITY//12)
    return librosa.midi_to_hz(f0_midi + midi_idx)


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


def bin_into_12(fine_pcd):
    kbw = KDE_GRANULARITY//12  # bin width
    s0 = kbw//2  # starting bin index
    
    pcd_12 = np.zeros(12)
    # 0-0.5 + 11.5-12
    pcd_12[0] = max(np.max(fine_pcd[:s0]), np.max(fine_pcd[s0 + 11 * kbw:]))
    # other bins
    for ii in range(11):
        pcd_12[ii+1] = np.max(fine_pcd[s0+ii*kbw:s0+(ii+1)*kbw])

    return pcd_12/np.sum(pcd_12)


def get_bin_support(note_index):
    """
    indices of the kde
    that correspond to the note_index
    """
    kbw = KDE_GRANULARITY//12  # bin width
    s0 = kbw//2  # starting bin index
    
    if note_index == 0:
        # 0-0.5 + 11.5-12
        bin_sup = np.concatenate((np.arange(s0 + 11 * kbw, KDE_GRANULARITY), np.arange(s0)))
    else:
        bin_sup = np.arange(s0+(note_index-1)*kbw, s0+note_index*kbw)

    return bin_sup
