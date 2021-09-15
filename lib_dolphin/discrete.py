import numpy as np

from numba import jit
from collections import namedtuple


Symbol = namedtuple("Symbol", "id type")


@jit
def levenstein(x, y):
    n = len(x)
    m = len(y)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(0, n + 1)
    d[0, :] = np.arange(0, m + 1)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            error = 0
            if x[i - 1] != y[j - 1]:
                error += 1
            d[i, j] = min([
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + error
            ])
    return d[n, m]


@jit
def levenstein_distances(X):
    N = len(X)
    d = np.zeros((N, N))
    for i in range(0, N):
        if i % 100 == 0:
            print(" ... distances {} / {} = {}".format(i, N, i / N))
        for j in range(i + 1, N):
            distance = levenstein(X[i], X[j])
            d[i, j] = distance
            d[j, i] = d[i, j]
    return d


def symbols(clusters, classifications):
    return [Symbol(c, l) for c, l in zip(clusters, classifications)]

