import numpy as np
from  sklearn.cluster import KMeans


def paa(double[:, :] sequence, int n): 
    '''
    Multi dimensional Piecewise Aggregate Approximation
    
    :params sequence: An nd sequence
    :params n: compress to length n
    :returns: compressed
    '''
    cdef int d = len(sequence[0])
    cdef int N = len(sequence)
    cdef int bucket = int(N / n)
    cdef int i = 0
    cdef double[:, :] compressed = np.zeros((n, d))
    if N < n:
        return np.array(sequence)
    for i in range(n):
        compressed[i] = np.mean(sequence[i * bucket : (i + 1) * bucket])
    return np.array(compressed)


def saxnd(list sequences, int n, int m):
    '''
    Multi dimensional Piecewise Aggregate Approximation
    
    :params sequences: A list of nd sequence
    :params n: compress to length n
    :params m: quantize to m symbols
    :returns: quantized sequence
    '''
    cdef int d = len(sequences[0][0])
    cdef int N = len(sequences)
    cdef int i, j = 0
    print("PAA")
    compressed = []
    for i in range(N):
        compressed.append(paa(sequences[i], n))
    compressed = np.vstack(compressed)
    print("Clustering")
    codebook = KMeans(n_clusters=m, n_init=10, max_iter=300)
    codebook.fit(compressed) 
    codes = []
    for i in range(N):
        code = []
        for j in range(len(compressed[i])):
            code.append(codebook.predict([compressed[i][j]]))
        print("Processing code: {}".format(code))
        codes.append(code)
    return codes


def similarity_bucketing(list sequences, int n, int m):
    '''
    Bueckting based on similarity
    
    :params sequences: A list of nd sequence
    :params n: compress to length n
    :params m: quantize to m symbols
    :returns: bucket id for each sequence
    '''
    cdef int N = len(sequences)
    cdef list codes = saxnd(sequences, n, m)
    cdef int[:] buckets = np.zeros(N, dtype=np.int32)
    cdef int i = 0
    cdef int cur = 0
    sequence_codebook = {}
    for i in range(N):
        key = tuple(list(codes[i]))
        if key not in sequence_codebook:
            sequence_codebook[key] = cur
            cur += 1
        buckets[i] = sequence_codebook[key]
    return np.array(buckets)
