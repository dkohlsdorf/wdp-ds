import numpy as np
from  sklearn.cluster import KMeans


def paa(float[:, :] sequence, int n): 
    cdef int d = len(sequence[0])
    cdef int N = len(sequence)
    cdef int bucket = int(N / n)
    cdef double[:, :] compressed = np.zeros((n, d))
    cdef int i = 0
    for i in range(n):
        compressed[i] = np.mean(sequence[i * bucket : (i + 1) * bucket])
    return np.array(compressed)


def saxnd(list sequences, int n, int m):
    cdef int d = len(sequences[0][0])
    cdef int N = len(sequences)
    cdef int i = 0
    compressed = []
    for i in range(N):
        compressed.append(paa(sequences[i], n))
    compressed = np.vstack(compressed)
    print(compressed.shape)
    codebook = KMeans(n_clusters=m, n_init=10, max_iter=10)   
    sequence = codebook.fit_predict(compressed) 
    print(sequence.shape)
    sequence = sequence.reshape((N, n))
    return np.array(sequence)


def similarity_bucketing(list sequences, int n, int m):
    cdef int N = len(sequences)
    cdef int[:, :] codes = saxnd(sequences, n, m)
    cdef int[:] buckets = np.zeros(N, dtype=np.int32)
    cdef int i = 0
    cdef int cur = 0
    sequence_codebook = {}
    for i in range(N):
        key = tuple(list(codes[i]))
        print(key)
        if key not in sequence_codebook:
            sequence_codebook[key] = cur
            cur += 1
        buckets[i] = sequence_codebook[key]
    return np.array(buckets)


def length_bucketing(list sequences, int n_bins, min_len = 1):
    cdef list lengths = [len(sequence) for sequence in sequences]
    cdef int n = len(lengths)
    cdef int max_len = max(lengths) 
    print(max_len)
    cdef int bucket_size = (max_len - min_len) / n_bins
    cdef int[:] buckets = np.zeros(n, dtype=np.int32)
    cdef int i = 0
    print(bucket_size)
    for i in range(n):
        if lengths[i] > min_len:
            buckets[i] = lengths[i] // bucket_size
    return np.array(buckets)