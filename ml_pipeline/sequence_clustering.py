import os
import multiprocessing as mp
import numpy as np
from dtw import *
from sklearn.cluster import AgglomerativeClustering
from sequence_hashing import similarity_bucketing

import logging
logging.basicConfig()
logcluster = logging.getLogger('cluster')
logcluster.setLevel(logging.INFO)    
                

def process_dtw(assignment, overlapping, max_dist, warping_band_percentage):
    """
    Cluster sequences
    :param assignment: Assignment id for bucket to cluster
    :param overlapping: All sequences
    :param max_dist: Distance for thresholding
    :returns: clustering, overlapping
    """
    n = len(overlapping)
    if n > 0:
        if n == 1:
            return [0], overlapping            
        # Compute distance matrix
        dist = np.zeros((n, n))
        for i, (start_x, stop_x, f_x, t_x, embedding_x) in enumerate(overlapping):
            if i % 250 == 0 and i > 0:
                logstructure.info("\t\t Processing: {} {}".format(i, len(overlapping)))            
            for j, (start_y, stop_y, f_y, t_y, embedding_y) in enumerate(overlapping):
                if i < j:
                    n  = len(embedding_x)
                    m  = len(embedding_y)
                    th = max_dist * n * m
                    w  = int(max(n, m) * warping_band_percentage)
                    d  = dtw(i, j, embedding_x, embedding_y, w)
                    dist[i, j] = d 
                    dist[j, i] = d 
        logcluster.info("\t {} {} {} {} {} {} ".format(assignment, n, np.percentile(dist.flatten(), 5), np.percentile(dist.flatten(), 95), np.mean(dist), np.std(dist)))
        
        # clustering
        agg = AgglomerativeClustering(n_clusters = None, distance_threshold = max_dist, linkage = 'complete', affinity='precomputed')
        clustering = agg.fit_predict(dist)
        return clustering, overlapping
    return [], []

                
def hc(overlapping, n_workers = 5, threshold = 0.5, warping=0.1, paa = 5, sax = 6):
    '''
    Hierarchical Clustering

    :param regions: the n regions to be clustered
    :param workers: how many parallel dtw computations are allowed
    :param threshold: frame distance threshold
    :param warping: percentage of allowed warping
    :return: array of n cluster ids
    '''
    sequences     = [np.stack(s) for _, _, _, _, s in overlapping]
    assignments   = similarity_bucketing(sequences, paa, sax)
    clusters      = max(assignments) + 1
    by_assignment = {}
    for o, s in zip(overlapping, assignments):
        if s not in by_assignment:
            by_assignment[s] = []
        by_assignment[s].append(o)
    
    logcluster.info("Bucketed Clustering")
    with mp.Pool(processes=n_workers) as pool:
        outputs = pool.starmap(process_dtw, ((assignment, overlapping, threshold, warping) for assignment, overlapping in by_assignment.items()))

    logcluster.info("Process Results")
    cur = 0
    assignments = []
    overlapping = []
    n_instances = {}
    for clustering, o in outputs:
        if len(clustering) > 0:
            for c, (start, stop, f, t, sequence) in zip(clustering, o):
                    if (c + cur) not in n_instances:
                        n_instances[c + cur] = 1
                    else:
                        n_instances[c + cur] += 1                                            
                    assignments.append(c + cur)
                    overlapping.append((start, stop, f, t, c + cur))
            for c in range(len(set(clustering))):
                cur += 1
    return overlapping


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
