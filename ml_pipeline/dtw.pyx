# Implements Dynamic Time Warping for Sequence Alignment
#
# REFERENCES:
# [HOL] John and Wendy Holmes: "Speech Synthesis and Recognition", Taylor & Francis Ltd; Second Edition, 2001
# [SAK] Sakoe, Chiba: "Dynamic programming algorithm optimization for spoken word recognition",  IEEE T. Acoust. Speech Signal Process. 26, 43–49, 1978.

import numpy as np
import matplotlib.pyplot as plt


def argmin(i, j, double deletion, double match, double insert):
    """
    Minimum of insertion / deletion or match at position i, j
    during dynamic time warping.

    :param i: i-th position in sequence x
    :param j: j-th position in sequence y
    :param deletion: alignment score for deletion
    :param match: alignment score for match
    :param insert: alignment score for insert
    :returns: minimum score and result index
    """
    result  = match
    res_idx = (i - 1, j - 1)
    if deletion < result:
        result  = deletion
        res_idx = (i - 1, j)
    if insert < result:
        result = insert
        res_idx = (i, j - 1) 
    return result, res_idx

PERCENTAGE_BAND = 10

cdef class DTW:

    cdef:
      cdef double[:,:] dp
      cdef int[:,:,:] bp
      cdef int band

    def __cinit__(self, int max_len):
        self.dp = np.ones((max_len + 1, max_len + 1)) * float('inf')      # Dynamic Programming Matrix 
        self.bp = np.zeros((max_len + 1, max_len + 1, 2), dtype=np.int32) # Back tracking matrix
        self.band = max_len // PERCENTAGE_BAND

    def align(self, double[:, :] x, double[:,:] y):
        """
        Align two sequences using dynamic time warping.

        Implements [HOL] equation 8.2

        :param x: a sequence of length N and dimension d
        :param y: a sequence of length M and dimension d
        :returns: alignment score and path
        """
        cdef unsigned int N = x.shape[0]
        cdef unsigned int M = y.shape[0]
        cdef int i, j = 0
        cdef double dist
        cdef int w = int(max(self.band, abs(N - M) + 2))
        cdef list path = []
        self.dp = np.multiply(self.dp, float('inf'))
        self.dp[0, 0] = 0.0
        for i in range(1, N + 1):
            for j in range(int(max(1, i - w)), int(min(M + 1, i + w))):
                dist = np.sum(np.square(np.subtract(x[i-1,:], y[j-1,:])))
                bp, (_i,_j) = argmin(i,j,self.dp[i - 1, j],self.dp[i - 1, j - 1], self.dp[i, j - 1])
                self.dp[i, j] = bp + dist
                self.bp[i, j, 0] = _i
                self.bp[i, j, 1] = _j
        i = N
        j = M
        while i > 0 and j > 0:
            path.append((i, j))
            _i = self.bp[i, j, 0]
            _j = self.bp[i, j, 1]
            i = _i
            j = _j
        path = list(reversed(path))
        return self.dp[N, M], path