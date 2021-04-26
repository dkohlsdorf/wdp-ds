import numba
import numpy as np


from numba import jit


@jit(nopython=True)
def is_interest_point(spectrogram, t, f, radius, threshold, use_th=True):
    neighborhood_time = np.sum(spectrogram[t - radius : t + radius, f])
    neighborhood_freq = np.sum(spectrogram[t, f - radius : f + radius])
    max_time          = np.max(spectrogram[t - radius : t + radius, f])
    max_freq          = np.max(spectrogram[t, f - radius : f + radius])
    noise             = (1.0 / (2.0 * radius)) * min(neighborhood_time, neighborhood_freq)
    local_max         = spectrogram[t,f] >= max(max_time, max_freq) 
    high_energy       = spectrogram[t,f] >= noise + threshold
    if use_th:
        return local_max and high_energy 
    else:
        return local_max or high_energy

    
@jit(nopython=True)
def interest_points(spectrogram, radius, threshold):
    T, F = spectrogram.shape
    for t in range(radius, T - radius):
        for f in range(radius, F - radius):
            if is_interest_point(spectrogram, t, f, radius, threshold):
                yield t, f
               
