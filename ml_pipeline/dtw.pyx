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


GAP = -1


def dtw(double[:, :] x, double[:,:] y, int band, double gap_penalty):
    """
    Align two sequences using dynamic time warping.

    Implements [HOL] equation 8.2
    :param x: a sequence of length N and dimension d
    :param y: a sequence of length M and dimension d
    :param band: sakoe chiba band
    :param gap_penalty: add offset to gaps
    :returns: alignment score, sequence of matches (i, j)
    """
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int w = max(band, abs(N - M)) + 2
    cdef int i, j = 0
    cdef double dist
    
    cdef double[:,:] dp         = np.ones((N + 1, M + 1)) * float('inf')
    cdef int[:,:,:]  bp         = np.zeros((N + 1, M + 1, 2), dtype=np.int32)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(max(1, i - w), min(M + 1, i + w)): # sakoe chiba band
            dist = np.sum(np.square(np.subtract(x[i-1,:], y[j-1,:]))) 
            hypothesis = [
                (i - 1, j,     dp[i - 1, j]     + dist + gap_penalty),
                (i - 1, j - 1, dp[i - 1, j - 1] + dist),
                (i, j - 1,     dp[i, j - 1]     + dist + gap_penalty)
            ]            
            _i, _j, d = min(hypothesis, key=lambda x: x[-1]) 
            dp[i][j]         = d
            bp[i][j][0]      = _i
            bp[i][j][1]      = _j

    i = N
    j = M
    matches = []
    while i > 0 and j > 0:
        _i, _j  = bp[i, j]        
        if i - 1 == _i and j - 1 == _j:
            matches.append((_i, _j))       
        i = _i
        j = _j
    return matches