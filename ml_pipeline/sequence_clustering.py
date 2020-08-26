import multiprocessing as mp
import numpy as np
from dtw import *


def distance_compute_job(regions, distance_threshold_frame, warping_band_percentage):
    '''
    Iterator for distance computation jobs

    :param regions: regions to be clustered
    :param distance_threshold_frame: threshold on each frame
    :param warping_band_percentage: percentage of warping
    :return: iterator over dtw job (i, j, region_i, region_j, th, w)
    '''
    for i, ri in enumerate(regions):    
        for j, rj in enumerate(regions):
            if i < j:
                n = len(ri)
                m = len(rj)
                th = distance_threshold_frame * n * m
                w  = int(max(n, m) * warping_band_percentage)
                yield i, j, ri, rj, th, w


def linkage(cluster_i, cluster_j, assignment, distances):
    '''
    Compute the average linkage between cluster i and cluster j
    :param cluster_i: ith cluster id
    :param cluster_j: jth cluster id
    :param assignment: array with cluster assignments
    :param distances: map[(i,j)] = distance
    :return: average linkage
    '''
    n = len(assignment)
    distance = 0.0
    size_x   = 0.0
    size_y   = 0.0
    for i in range(n):
        if assignment[i] == cluster_i:
            for j in range(n):
                if assignment[j] == cluster_j:
                    if (i, j) in distances:
                        distance += distances[(i,j)]
                    size_y += 1.0
            size_x += 1.0
    return distance / (size_x * size_y)


def hc(regions, n_workers = 5, threshold = 15.0, warping=0.1):
    '''
    Hierarchical Clustering

    :param regions: the n regions to be clustered
    :param workers: how many parallel dtw computations are allowed
    :param threshold: frame distance threshold
    :param warping: percentage of allowed warping
    :return: array of n cluster ids
    '''
    with mp.Pool(processes=n_workers) as pool:
        result = pool.starmap(write_audio, (distance_compute_job(regions, threshold, warping)))
    sparse_dist = dict([((i, j), d) for i, j, d in results if not np.isinf(d)])
    n = len(regions)
    assignment = np.arange(n)
    min_linkage = 0.0
    while min_linkage < threshold:
        min_linkage = float('inf')
        for cluster_i in set(assignment):
            for cluster_j in set(assignment):
                if cluster_i != cluster_j:
                    l = linkage(cluster_i, cluster_j, assignment, sparse_dist)
                    if l < min_linkage:
                        min_linkage = l
        if i < j:
            assignment[j] = i  
        else:
            assignment[i] = i
    return assignment


def overlap(x1, x2):
    """
    Do two regions (x1_start, x1_stop, file, type) and (x2_start, x2_stop, file, type) overlap?

    :param x1: first region tuple 
    :param x2: second reggion tuple
    """
    return max(x1[0],x2[0]) <= min(x1[1],x2[1]) and x1[2] == x2[2] and x1[3] == x2[3]


def mk_region(sequences):
    """
    Convert a set of grouped sequences into a region
    
    :param sequences: [
        [(start, stop, file, type, x), (start, stop, file, type, x)], 
        [(start, stop, file, x)], 
        [(start, stop, file, x), (start, stop, file, x)]]
    :returns: [([x, x], start, stop), ([x], start, stop),([x, x], start, stop)]
    """
    for x in sequences:
        items = [item for _, _, _, _, item in x]
        start = x[0][0]
        stop  = x[-1][1]
        f     = x[-1][2]
        t     = x[-1][3]
        yield start, stop, f, t, items


def groupBy(sequences, grouping_cond, window_size=None):
    """
    Extract groups of signals
    :param sequences: a sortd list of time stamped embedding sequences (seq, start, stop, file, type)
    :param grouping_cond: a function (x1, x2) returning true if two items should be grouped
    :returns: grouped sequence list
    """
    groups = []
    current = []
    for x in sequences:
        if len(current) == 0:
            current.append(x)
        else:
            if grouping_cond(current[-1], x):
                current.append(x)
                if window_size is not None and window_size == len(current):
                    groups.append(current)
                    current = current[1:]
            else:
                if window_size is None:
                    groups.append(current)
                    current = [x]
    if len(current) > 0:
        groups.append(current)
    return [x for x in mk_region(groups)]
