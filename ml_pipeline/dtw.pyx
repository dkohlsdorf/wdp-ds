# Implements Dynamic Time Warping for Sequence Alignment
#
# REFERENCES:
# [HOL] John and Wendy Holmes: "Speech Synthesis and Recognition", Taylor & Francis Ltd; Second Edition, 2001
# [SAK] Sakoe, Chiba: "Dynamic programming algorithm optimization for spoken word recognition",  IEEE T. Acoust. Speech Signal Process. 26, 43–49, 1978.
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.basicConfig()
logdtw = logging.getLogger('dtw')
logdtw.setLevel(logging.INFO)


def dtw(int id_x, int id_y, double[:, :] x, double[:,:] y, int band):
    """
    Align two sequences using dynamic time warping.

    Implements [HOL] equation 8.2
    :param id_x: position in all sequences
    :param id_y: position in all sequences 
    :param x: a sequence of length N and dimension d
    :param y: a sequence of length M and dimension d
    :param band: sakoe chiba band
    :returns: id_x, id_y, alignment score
    """
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int w = max(band, abs(N - M)) + 2
    cdef int i, j = 0
    cdef double dist
    cdef double[:,:] dp = np.ones((N + 1, M + 1)) * float('inf')
    
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(max(1, i - w), min(M + 1, i + w)): # sakoe chiba band
            v    = min([dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]]) 
            dist = np.sum(np.square(np.subtract(x[i-1,:], y[j-1,:]))) 
            dp[i][j] = v + dist
											
    score = dp[N, M] / (N * M)
    if i % 100 == 0:
        logdtw.info("DTW({} {} | {}) = {} ".format(id_x, id_y, w, score))
    return id_x, id_y, score