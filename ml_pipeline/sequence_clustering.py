import os
import multiprocessing as mp
import numpy as np
from dtw import *
from sklearn.cluster import AgglomerativeClustering

import logging
logging.basicConfig()
logcluster = logging.getLogger('cluster')
logcluster.setLevel(logging.INFO)


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
                yield i, j, ri, rj, w


def dtw_process(i, j, ri, rj, w):
    if i % 100 == 0:
        logcluster.info("Process dtw({}, {})".format(i, j, w))
    return dtw(i,j, np.stack(ri), np.stack(rj), w)
    
                
def hc(regions, out, n_workers = 5, threshold = 0.5, warping=0.1):
    '''
    Hierarchical Clustering

    :param regions: the n regions to be clustered
    :param workers: how many parallel dtw computations are allowed
    :param threshold: frame distance threshold
    :param warping: percentage of allowed warping
    :return: array of n cluster ids
    '''
    logcluster.info("Start clustering distance precompute with {} workers".format(n_workers))
    if os.path.isfile('{}/distances.npy'.format(out)):
        logcluster.info("Loading Precomputed Distances")
        distances = np.load('{}/distances.npy'.format(out))
    else:
        n = len(regions)
        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(dtw_process, (distance_compute_job(regions, threshold, warping)))
        logcluster.info("Done distance computation for {} instances".format(n))
        distances = np.zeros((n, n))
        for i, j, d in results:
            distances[i, j] = d
            distances[j, i] = d
        np.save('{}/distances.npy'.format(out), distances)
        
    logcluster.info("Done writing distances: p95 = {}, p1 = {}, p5 = {}, median = {}".format(
        np.percentile(distances, 95),
        np.percentile(distances, 1),
        np.percentile(distances, 5),
        np.percentile(distances, 50)
    ))
    agg = AgglomerativeClustering(n_clusters = None, distance_threshold = threshold, linkage = 'complete', affinity='precomputed')
    assignment = agg.fit_predict(distances)
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


def groupBy(sequences, grouping_cond, min_len):
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
            else:
                groups.append(current)
                current = [x]
    if len(current) > 0:
        groups.append(current)
    return [x for x in mk_region(groups) if len(x[4]) >= min_len]
