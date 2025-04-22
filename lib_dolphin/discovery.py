import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from fastdtw import fastdtw
from copy import deepcopy
from collections import defaultdict


def pairwise_dtw_distance_matrix(sequences):
    N = len(sequences)
    D = np.zeros((N * (N - 1)) // 2)
    k = 0
    total = len(D)
    for i in range(N):
        for j in range(i + 1, N):
            if k % 10000 == 0:
                percentage = k / total
                print(f"DTW distances: {percentage * 100}")
            dist, path = fastdtw(sequences[i], sequences[j], dist=lambda x, y: np.linalg.norm(x - y))
            D[k] = dist / len(path)
            k += 1
    return D


def hierarchical_clustering(distances, th):
    Z = linkage(distances, method='complete')
    labels = fcluster(Z, t=th, criterion='distance')
    return labels


def dtw_barycenter_avg(sequences, max_iters=10):
    lengths = [len(s) for s in sequences]
    median_len = sorted(lengths)[len(lengths) // 2]
    init = min(sequences, key=lambda x: abs(len(x) - median_len))
    bary = deepcopy(init)
    for it in range(max_iters):
        assignments = [[] for _ in range(len(bary))]

        for seq in sequences:
            _, path = fastdtw(seq, bary, dist=lambda x, y: np.linalg.norm(x - y))
            for i, j in path:
                assignments[j].append(seq[i])

        new_bary = []
        variances = []
        for group in assignments:
            if group:
                new_bary.append(np.mean(group, axis=0))
                variances.append(np.mean(np.var(group, axis=0)))
        bary = new_bary
    return bary, variances


def extract_alignment_points(clustered_sequences, barycenters, instance_ids,
                             variance_th=0.1, min_count=5):
    reject_count = 0
    reject_var = 0
    alignment_points = {}
    for cluster_id, sequences in clustered_sequences.items():
        instances = instance_ids[cluster_id]
        bary = barycenters[cluster_id]
        point_alignments = defaultdict(list) 
        for seq, instance in zip(sequences, instances):
            _, path = fastdtw(seq, bary, dist=lambda x, y: np.linalg.norm(x - y))
            for seq_idx, bary_idx in path:
                point_alignments[bary_idx].append((seq[seq_idx], instance, seq_idx))
        
        tight_points = defaultdict(list)
        for bary_idx, items in point_alignments.items():
            if len(items) >= min_count:                
                vectors = [v for v, _, _ in items]
                ids = [(i, j) for _, i, j in items]
                stack = np.stack(vectors)
                var = np.mean(np.var(stack, axis=0))
                if var <= variance_th:
                    tight_points[bary_idx] = ids
                else:
                    reject_var += 1
            else:
                reject_count += 1
        alignment_points[cluster_id] = tight_points
    print(f"\tCount: {reject_count} Var: {reject_var}")
    return alignment_points
