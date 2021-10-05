import numpy as np


def distances(X, Y):
    return -2.0 * np.dot(X, X.T)\
        + np.sum(X ** 2, axis=1)\
        + np.sum(Y ** 2, axis=1)[:, np.newaxis]


def knn(X, Y, k):
    d = distances(X, Y)
    n = d.shape[0]
    densities = []
    neighbors = []
    for i, x in enumerate(d):
        x = [(j, dist) for j, dist in enumerate(x)]
        x = sorted(x, key=lambda x: x[-1])[:self.k]                
        x_density = 1. / (x[-2][0] + 1e-8)
        densities.append(x_density)
        neighbors.append([i for i, _ in x])
    return densities, neighbors

        
class DensityBasedDiscovery:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        densities, neighbors = knn(X, X, self.k)
        max_densities = []
        for i, density in enumerate(densities):
            max_density = True
            for j in neighbors[i]:
                if density <= densities[j]:
                    max_density = False
            if max_density:
                max_densities.append(i)

        centroids = X[max_densities, :]            
        self.centroids = centroids

    def predict(self, X):
        _, neighbors = knn(X, self.centroids, 1)
        return [nn[0] for nn in neighbors]

        

