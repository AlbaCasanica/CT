import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz, welch
from scipy.fft import fft, ifft
from scipy import stats
# from scipy import signal
from math import log, e
import numpy as np
import pandas as pd

"This file is based on the bachelor students modification of Royas code"

def get_Freq_Domain_features_of_signal(signal, signal_name, Fs):

    features = {} 

    suffix = ""
    if signal_name:
        suffix = f"_{signal_name}"
    
    signal_t = np.transpose(signal)
     
    # Compute PSD via Welch algorithm
    # plot psd to compare different values of nperseg?
    freq, psd = welch(signal_t, Fs, nperseg=1024, scaling='density')  # az
    
    # Convert to [ug / sqrt(Hz)]
    psd = np.sqrt(psd) #* accel2ug

    # Compute noise spectral densities        
    # ndaz = np.mean(psd)


    features[f'psd_mean{suffix}']           = psd.mean()   
    features[f'psd_max{suffix}']            = psd.max()
    features[f'psd_min{suffix}']            = psd.min()
    features[f'psd_max(Hz)_{suffix}']       = freq[np.argmax(psd)] # have not found this uesd in any of the paper I have reads
    features[f'psd_energy{suffix}']         = sum(pow(abs(psd), 2)) # https://stackoverflow.com/questions/34133680/calculate-energy-of-time-domain-data
    features[f'psd_entropy_{suffix}']       = entropy(psd) # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    # features[f'Hz_mean{suffix}']            = freq.mean()  # This does not make sense

    return features


def entropy(labels, base=None):
    """ Computes entropy of label distribution. """
    #https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent