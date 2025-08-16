
from math import log, e
import numpy as np
from scipy import stats

"This file is based on the bachelor students modification of Royas code"

def get_Time_Domain_features_of_signal(signal, signal_name, epsilon=1e-8):
    features = {} 

    suffix = ""
    if signal_name:
        suffix = f"_{signal_name}"
    
    features[f'mean{suffix}']       = signal.mean()   
    features[f'sd{suffix}']         = signal.std()
    features[f'mad{suffix}']        = stats.median_abs_deviation(signal, scale=1/1.4826)
    features[f'max{suffix}']        = signal.max()
    features[f'min{suffix}']        = signal.min()
    features[f'energy{suffix}']     = sum(pow(abs(signal), 2)) # https://stackoverflow.com/questions/34133680/calculate-energy-of-time-domain-data
    features[f'entropy{suffix}']    = entropy(signal) # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    features[f'iqr{suffix}']        = interquartile_range(signal) # https://www.statology.org/interquartile-range-python/
    
    if features[f'sd{suffix}'] < epsilon:
        features[f'skewness{suffix}']   = 0.0
        features[f'kurtosis{suffix}']   = 0.0  # Fisher-normal
    else:
        features[f'skewness{suffix}']   = stats.skew(signal)
        features[f'kurtosis{suffix}']   = stats.kurtosis(signal, fisher=True)

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


def interquartile_range(signal):
    """finds the interquartile range of a 1d numpy array"""
    # https://www.statology.org/interquartile-range-python/

    q3, q1 = np.percentile(signal, [75, 25])
    iqr = q3 - q1
    return iqr
