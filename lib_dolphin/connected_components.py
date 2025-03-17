import numpy as np


def find(p, ids):
    while p < len(ids) and p != ids[p]:
        p = ids[p]
    return p


def union(p, q, ids, region_sizes):
    i = find(p, ids)
    j = find(q, ids)
    if i != j:
        if region_sizes[i] < region_sizes[j]:
            ids[i] = ids[j]
            region_sizes[j] += region_sizes[i] 
        else:
            ids[j] = ids[i]
            region_sizes[i] += region_sizes[j] 
    
    
def connected_components(sequence, detection_th, window_size, min_region_size):
    x = sequence.rolling(window=window_size).mean()
    x[0:window_size] = 0
    x = x > detection_th
    n = len(x)
    region_sizes = np.ones(len(x))
    ids = np.arange(len(x), dtype=np.int32)
    for i in range(1, len(x)):
        if x[i] and x[i - 1]:
            union(i-1, i, ids, region_sizes)
    for i in ids:
        p = find(i, ids)
        if region_sizes[p] < min_region_size:
            ids[i] = 0
    return ids
