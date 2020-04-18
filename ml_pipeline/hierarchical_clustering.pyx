import numpy as np

cdef class Agglomerative:
    cdef:
        cdef int[:] assignment
        cdef double th
        cdef int n

    def __init__(self, distances, double th):
        (_, n) = distances.shape
        self.assignment = np.zeros(n, dtype=np.int32)
        self.distances  = distances
        self.n          = n
        self.th         = th

    def merge(self, cluster_i, cluster_j):
        if cluster_i > cluster_j:
            tmp = cluster_i
            cluster_i = cluster_j
            cluster_j = tmp
        cdef unsigned int i = 0
        for i in range(0, self.n):
            if self.assignment[i] == cluster_j:
                self.assignment[i] = cluster_i

    def upgma(self, int cluster_i, int cluster_j):
        cdef int i,j = 0
        cdef int n = self.n
        cdef double dist = 0.0
        for i in range(0, n):
            for j in range(0, n):
                if self.assignment[i] == cluster_i and self.assignment[j] == cluster_j:
                    dist += self.distances[i, j]
        cdef int num_i, num_j = 0
        for i in range(0, n):
            if self.assignment[i] == cluster_i:
                num_i += 1
        for j in range(0, n):
            if self.assignment[j] == cluster_j:
                num_j += 1
        dist /= (num_i, num_j)
        return dist

    def best_merge(self):
        cdef int[:] clusters = np.unique(self.assignment)
        cdef int n = len(clusters)
        cdef int i,j = 0
        cdef double dist = 0.0
        cdef double min_value = float('inf')
        cdef int min_cluster_i = 0
        cdef int min_cluster_j = 0
        for i in range(0, n):
            for j in range(0, i):
                dist = self.upgma(clusters[i], clusters[j])
                if i != j and dist > 0.0 and dist < min_value:
                    min_value = dist
                    min_cluster_i = clusters[i]
                    min_cluster_j = clusters[j]
        if min_value < self.th:
            self.merge(min_cluster_i, min_cluster_j)
        return min_value

    def cluster(self):
        cdef int n = len(self.assignment)
        cdef double dist = 0.0
        while n > 1:
            dist = self.best_merge()
            if dist >= self.th:
                break
            n = len(np.unique(self.assignment))
    