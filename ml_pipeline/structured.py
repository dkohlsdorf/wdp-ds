import numpy as np
import pandas as pd
import os 
import pickle as pkl
import tensorflow as tf
import re
import multiprocessing as mp

from scipy.sparse import lil_matrix

from pomegranate import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import namedtuple
from dtw import DTW
from sequence_hashing import similarity_bucketing
from sklearn.cluster import AgglomerativeClustering


class TypeExtraction(namedtuple("Induction", "embeddings starts stops types files")):
    """
    Type annotations for dolphin communication   
    """

    @property
    def len(self):
        return len(self.starts)

    def items(self):        
        """
        Iterate annotated tuples (filename, start, stop, type, embedding vector)
        """
        n = self.len
        for i in range(n):
            yield self.files[i], self.starts[i], self.stops[i], self.types[i], self.embeddings[i]

    @classmethod
    def from_audiofile(cls, path, embedder):
        """
        Construct type annotations from folder with audio files

        :param folder: folder with audio files
        :param embedder: a sequence embedder
        :returns: type annotation
        """
        embeddings = []
        starts     = []
        stops      = []
        types      = []
        files      = [] 
        print("- Working on embedding {}".format(path))
        regions = embedder.embed(path)
        for x, f, start, stop, t in regions:
            embeddings.append(x)
            starts.append(start)
            stops.append(stop)
            types.append(t)
            files.append(f)
        return cls(embeddings, starts, stops, types, files) 

    def save(self, path, append=False):
        mode = "w"
        if append:
            mode += "+"
        with open(path, mode) as fp:
            for filename, start, stop, t, embedding in self.items():
                csv = ','.join(['%.5f' % f for f in embedding])
                fp.write("{}\t{}\t{}\t{}\t{}\n".format(filename, start, stop, t, csv))


def overlap(x1, x2):
    '''
    Do two regions (x1_start, x1_stop, file, type) and (x2_start, x2_stop, file, type) overlap?

    :param x1: first region tuple 
    :param x2: second reggion tuple
    '''
    return max(x1[0],x2[0]) <= min(x1[1],x2[1]) and x1[2] == x2[2] and x1[3] == x2[3]


def mk_region(sequences):
    '''
    Convert a set of grouped sequences into a region

    :param sequences: [
        [(start, stop, file, type, x), (start, stop, file, type, x)], 
        [(start, stop, file, x)], 
        [(start, stop, file, x), (start, stop, file, x)]]
    :returns: [([x, x], start, stop), ([x], start, stop),([x, x], start, stop)]
    '''
    for x in sequences:
        items = [item for _, _, _, _, item in x]
        start = x[0][0]
        stop  = x[-1][1]
        f     = x[-1][2]
        t     = x[-1][3]
        yield start, stop, f, t, items


def groupBy(sequences, grouping_cond, window_size=None):
    '''
    Extract groups of signals

    :param sequences: a sortd list of time stamped embedding sequences (seq, start, stop, file, type)
    :param grouping_cond: a function (x1, x2) returning true if two items should be grouped
    :returns: grouped sequence list
    '''
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
    return [x for x in mk_region(groups)]


def process_dtw(assignment, overlapping, max_dist):
    '''
    Cluster sequences

    :param assignment: Assignment id for bucket to cluster
    :param overlapping: All sequences
    :param max_dist: Distance for thresholding
    :returns: clustering and overlapping
    '''
    n = len(overlapping)
    if n > 1:
        max_len = int(max([len(e) for _, _, _, _, e in overlapping]) + 1)
        dtw = DTW(max_len)
        dist = np.zeros((n, n))
        for i, (start_x, stop_x, f_x, t_x, embedding_x) in enumerate(overlapping):
            if i % 250 == 0 and i > 0:
                print("\t\t Processing: {} {}".format(i, len(overlapping)))
            for j, (start_y, stop_y, f_y, t_y, embedding_y) in enumerate(overlapping):
                if i < j:
                    x = np.array([embedding_x]).reshape(len(embedding_x), 256)
                    y = np.array([embedding_y]).reshape(len(embedding_y), 256)
                    d, _       = dtw.align(x, y) 
                    dist[i, j] = d / (len(x) * len(y))
                    dist[j, i] = d / (len(x) * len(y))
        print("\t {} {} {} {} {} {} ".format(assignment, n, np.percentile(dist.flatten(), 5), np.percentile(dist.flatten(), 95), np.mean(dist), np.std(dist)))
        agg = AgglomerativeClustering(n_clusters = None, 
                                      distance_threshold = max_dist, linkage = 'average', affinity='precomputed')
        clustering = agg.fit_predict(dist)
        return clustering, overlapping
    return [], []


def make_hmm(cluster, assignment, overlapping, min_instances = 5):
    '''
    Learn a 4 state Hidden Markov Model with 2 skip states.
    Initialization is performed from using flat start (mean and variances equal for all states)
    Training is performed using Baum Welch

    :param cluster: cluster number
    :param assignment: assignment of clusters for each sequence
    :param overlapping: the overlapping sequences
    :returns: a hidden markov model   
    '''
    x_label = np.array([overlapping[i] for i in range(0, len(overlapping)) if assignment[i] == cluster])
    if len(x_label > min_instances):
        frames = int(np.mean([len(x) for x in x_label]))
        n = frames / 4
        l = 1 / n
        s = 1 - l
        trans_mat = numpy.array([[s,     l/2, l/2, 0.0],
                                [0.0,   s, l,     0.0],
                                [0.0, 0.0, s,       l],
                                [0.0, 0.0, 0.0,     s]])
        starts    = numpy.array([1.0, 0.0, 0.0, 0.0])
        ends      = numpy.array([0.0, 0.0, 0.0, 0.1])
        dim       = len(overlapping[0][0])
        dists     = []
        for i in range(0, 4):
            state = x_label[0][i * int(n): (i + 1) * int(n)]
            mu    = np.mean(state, axis=(0, 1))
            std   = np.eye(dim) * (np.std(state, axis=(0, 1)) + 1.0)
            dists.append(MultivariateGaussianDistribution(mu, std))
        model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
        model.fit(x_label, algorithm='baum-welch', max_iterations=50, verbose=True)
        print(model)
        return model
    return None


def decode(sequence, hmms):
    '''
    Decode all sequences using a hidden Markov model
    :param sequence: a  sequences to decode
    :param hmms: a list of hidden markov model
    :returns: (max likelihoods, max assignment)
    '''
    max_ll = 0.0
    max_hmm     = 0
    for i, hmm in enumerate(hmms):
        ll = hmm.log_probability(sequence)
        if ll > max_ll:
            max_ll = ll
            max_hmm = i
    return max_ll, max_hmm


def greedy_mixture_learning(sequences, hmms, th):
    '''
    Greedily learn a mixture of hidden markov models

    :param sequences: a list of sequences
    :param hmms: a list of hidden Markov models
    :param pool: a thread pool
    :param th: stop when improvement is below a threshold
    :returns: final set of hmms 
    '''
    last_ll = float('-inf')
    models   = []
    openlist = hmms.copy()
    while len(openlist) > 0:
        max_hypothesis_ll = float('-inf')
        max_hypothesis    = 0
        for i, hmm in enumerate(openlist):
            hypothesis = models + [hmm]
            pool = mp.Pool(processes=processes)
            results     = [pool.apply_async(decode, args=(sequence, hypothesis)) for sequence in sequences]
            decoded     = [p.get() for p in results]
            pool.close()
            assignemnts = [assignment for _, assignment in decoded]
            likelihoods = [ll for ll, _ in decoded]
            likelihood   = sum(likelihoods)
            if likelihood > max_hypothesis_ll:
                max_hypothesis_ll = likelihood
                max_hypothesis = i
        best   = openlist.pop(max_hypothesis)
        models = models + [best]
        print("Greedy Mixture Learning: {}".format(max_hypothesis_ll))
        if last_ll - max_hypothesis_ll < th:
            break
    pool        = mp.Pool(processes=processes)
    results     = [pool.apply_async(decode, args=(sequence, hmms)) for sequence in sequences]
    decoded     = [p.get() for p in results]
    assignemnts = [assignment for _, assignment in decoded]
    pool.close()
    return models, last_ll, assignemnts


def hierarchical_clustering(
    annotation_path,
    max_dist = 10.0, 
    min_th= 1, 
    max_th= 500, 
    paa = 3, 
    sax = 4,
    processes = 25,
    max_instances=None
):
    '''
    Hierarchical clustering of annotations
    :param annotation_path: path to work folder
    :param max_dist: distance threshold for clustering
    :param min_len: minimum length of sequence
    :param max_len: maximum length of sequence
    :param paa: compressed size
    :param sax: quantization codebook size
    :param processes: number of threads
    :returns: clustering result [(start, stop, filename, cluster)]
    '''
    overlapping           = []
    for file in tf.io.gfile.listdir(annotation_path):        
        if file.startswith("embedding") and file.endswith(".csv"):
            path = "{}/{}".format(annotation_path, file)
            print("\tReading {}".format(path))
            header                = ["filename", "start", "stop", "type", "embedding"]
            df                    = pd.read_csv(path, sep="\t", header = None, names=header)
            signals               = df[df['type'] > 1]
            signals['embedding']  = df['embedding'].apply(
                lambda x: np.array([float(i) for i in x.split(",")]))
            annotated             = [(row['start'], row['stop'], row['filename'], row['type'], row['embedding'])
                                     for _ , row in signals.iterrows()]
            overlapping += groupBy(annotated, overlap)
            if max_instances is not None and len(overlapping) > max_instances:
                break
                
    overlapping = [x for x in overlapping if len(x[4]) > min_th and len(x[4]) < max_th]
    max_len = int(max([len(e) for _, _, _, _, e in overlapping]) + 1)
    sequences = [np.stack(s) for _, _, _, _, s in overlapping]
    print("Bucketing instances")
    if max_instances is not None:
        assignments = similarity_bucketing(sequences, paa, sax, max_instances)
    else:
        assignments = similarity_bucketing(sequences, paa, sax)
    clusters = max(assignments) + 1
    by_assignment = {}
    for o, s in zip(overlapping, assignments):
        if s not in by_assignment:
            by_assignment[s] = []
        by_assignment[s].append(o)
    print("Bucketed Clustering")
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(process_dtw, args=(assignment, overlapping, max_dist)) for assignment, overlapping in by_assignment.items()]
    outputs = [p.get() for p in results]
    pool.close()

    print("Process Results")
    cur = 0
    assignments = []
    overlapping = []
    sequences   = []
    for clustering, overlapping in outputs:
        if len(clustering) > 0:
            for c, (start, stop, f, t, sequence) in zip(clustering, overlapping):
                    assignments.append(c)
                    overlapping.append((start, stop, f, t, c + cur))
                    sequences.append(sequence)
            for c in range(len(set(clustering))):
                cur += 1
    print("Build Hidden Markov Models")
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(make_hmm, args=(c, assignments, sequences)) for c in range(len(assignments))]
    hmms    = [p.get() for p in results]
    hmms    = [hmm for hmm in hmms if hmm is not None]
    pool.close()
    print("Greedy Mixture Learning / Cluster Supression")
    models, last_ll, assignemnts = greedy_mixture_learning(sequences, hmms, 1e-6)
    cluster_regions = [(start, stop, f, t, c) for c, (start, stop, f, t, _) in zip(assignments, overlapping)]
    return cluster_regions


def annotate_clustering(work_folder, annotations):
    '''
    Annotates a clustering

    :param work_folder: folder with clustering results
    :param annotations: file with annotations
    :returns: dict[clusters][filename][start, stop, annotation]
    '''
    header = ["cluster", "type"]
    df = pd.read_csv(annotations, sep=",", header = None, names=header)    
    annotations = {}
    for i, row in df.iterrows():
        c = int(row['cluster'])
        annotations[c] = row['type']
        
    clusters = {}    
    for file in tf.io.gfile.listdir(work_folder):
        if file.startswith("seq_clustering") and file.endswith(".csv"):        
            path = "{}/{}".format(work_folder, file)
            df = pd.read_csv(path)            
            for i, row in df.iterrows():
                c = int(row['cluster'])
                filename = row['file']
                start = row['start']
                stop = row['stop']

                if c in annotations:
                    if c not in clusters:
                        clusters[c] = {}
                    if filename not in clusters[c]:
                        clusters[c][filename] = []
                    clusters[c][filename].append((start, stop, annotations[c]))
    return clusters
