import numpy as np

def argmin(i, j, double deletion, double match, double insert):
    result  = match
    res_idx = (i - 1, j - 1)
    if deletion < result:
        result  = deletion
        res_idx = (i - 1, j)
    if insert < result:
        result = insert
        res_idx = (i, j - 1) 
    return result, res_idx

cdef class DTW:

    cdef:
      cdef double[:,:] dp
      cdef int[:,:,:] bp

    def __cinit__(self, int max_len):
        self.dp = np.ones((max_len + 1, max_len + 1)) * float('inf')
        self.bp = np.zeros((max_len + 1, max_len + 1, 2), dtype=np.int32)

    def align(self, double[:, :] x, double[:,:] y):
        cdef unsigned int N = x.shape[0]
        cdef unsigned int M = y.shape[0]
        cdef unsigned int i, j = 0
        cdef double dist;
        cdef list path = []
        self.dp = np.multiply(self.dp, float('inf'))
        self.dp[0, 0] = 0.0
        for i in range(1, N + 1):
            for j in range(1, M + 1):
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