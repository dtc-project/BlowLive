import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.fftpack
from gammatone.gtgram import gtgram
from python_speech_features import delta

# Parameters
sr = 48000               # Sampling rate
frame_len = 0.05         # 10 ms frame length
frame_step = 0.02       # 5 ms hop length
num_filters = 24         # Number of filters (wide enough for blowing signals)
num_ceps = 13            # First 13 coefficients
f_min = 40             # Minimum frequency for Gammatone filterbank
remove_energy = False     # Remove energy coefficient
include_delta2 = False
include_delta = True


def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def compute_gfcc(filename):
    # Load and preprocess
    sr, y = wavfile.read(filename)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128) / 128
    y = pre_emphasis(y)

    # Compute gammatonegram (time x filter)
    win_size = int(sr * frame_len)
    hop_size = int(sr * frame_step)
    gt = gtgram(y, sr, window_time=frame_len, hop_time=frame_step,
                channels=num_filters, f_min=f_min)

    # Convert to log
    log_gt = np.log(gt + 1e-10)

    # DCT to get cepstral coefficients
    gfcc = scipy.fftpack.dct(log_gt, type=2, axis=0, norm='ortho')[0:num_ceps, :]

    # Optional: remove 0th coefficient (energy)
    if remove_energy:
        gfcc = gfcc[1:, :]  # remove the 0th

    # Delta and Delta-Delta
    delta_gfcc = delta(gfcc.T, 2).T
    if include_delta2:
        delta2_gfcc = delta(delta_gfcc.T, 2).T
        combined = np.vstack((gfcc, delta_gfcc, delta2_gfcc))
    elif include_delta:
        combined = np.vstack((gfcc, delta_gfcc))
    else:
        combined = np.vstack((gfcc))
    return combined


import numpy as np

def save_gfcc_csv(gfcc_matrix, filename):
    np.savetxt(filename, gfcc_matrix.T, delimiter=',')  # Transpose for easier DTW later

