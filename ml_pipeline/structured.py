# Structuring Embedded Sequences
# 
# 1) Detect Overlapping Non-Noise Subsequences
# 2) Cluster these sequences
# 
# REFERENCES: 
# [MIN] Minnen, Isbel, Essa, Starner: "Discovering Multivariate Motifs using Subsequence Density Estimation and Greedy Mixture Learning", AAAI, 2007
# [KOH1] Daniel Kohlsdorf: "Data Mining In Large Audio Collections Of Dolphin Signals", Georgia Tech, Doctoral Thesis 2015
# [KOH2] Daniel Kohlsdorf, Denise Herzing, Thad Starner: "Feature Learning and Automatic Segmentation for Dolphin Communication Analysis", Interspeech16, 2016
# [KOH3] Daniel Kohlsdorf, Denise Herzing and Thad Starner: "Methods for Discovering Models of Behavior: A Case Study with Wild Atlantic Spotted Dolphins", Animal Behavior and Cognition, November 2016

import numpy as np
import pandas as pd
import os 
import pickle as pkl
import tensorflow as tf
import re
import multiprocessing as mp
import random

import logging
logging.basicConfig()
logstructure = logging.getLogger('structure')
logstructure.setLevel(logging.INFO)

from sklearn.cluster import AgglomerativeClustering

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import namedtuple
from dtw import DTW
from sequence_hashing import similarity_bucketing
from markov_chain import DenseMarkovChain, Transition, START_STATE, STOP_STATE
from logprob import LogProb, ZERO
from hidden_markov_model import HiddenMarkovModel
from viterbi import viterbi
from distributions import Gaussian

import fwd_bwd    as infer
import baum_welch as bw


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
        logstructure.info("- Working on embedding {}".format(path))
        regions = embedder.embed(path)
        logstructure.info("\t- found region {}".format(len(regions)))
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


def process_dtw(assignment, overlapping, max_dist):
    """
    Cluster sequences

    :param assignment: Assignment id for bucket to cluster
    :param overlapping: All sequences
    :param max_dist: Distance for thresholding
    :returns: clustering and overlapping
    """
    n = len(overlapping)
    if n > 1:
        max_len = int(max([len(e) for _, _, _, _, e in overlapping]) + 1)
        dtw = DTW(max_len)

        # Compute distance matrix
        dist = np.zeros((n, n))
        for i, (start_x, stop_x, f_x, t_x, embedding_x) in enumerate(overlapping):
            
            if i % 250 == 0 and i > 0:
                logstructure.info("\t\t Processing: {} {}".format(i, len(overlapping)))
            
            for j, (start_y, stop_y, f_y, t_y, embedding_y) in enumerate(overlapping):
                if i < j:
                    x = np.array([embedding_x]).reshape(len(embedding_x), 256)
                    y = np.array([embedding_y]).reshape(len(embedding_y), 256)
                    d, _       = dtw.align(x, y) 
                    dist[i, j] = d / (len(x) * len(y))
                    dist[j, i] = d / (len(x) * len(y))
        logstructure.info("\t {} {} {} {} {} {} ".format(assignment, n, np.percentile(dist.flatten(), 5), np.percentile(dist.flatten(), 95), np.mean(dist), np.std(dist)))
        
        # clustering
        agg = AgglomerativeClustering(n_clusters = None, 
                                      distance_threshold = max_dist, linkage = 'average', affinity='precomputed')
        clustering = agg.fit_predict(dist)
        return clustering, overlapping
    return [], []


def make_hmm(cluster, assignment, overlapping, min_len = 8, max_train=15):
    """
    Learn a 4 state Hidden Markov Model with 2 skip states.
    Initialization is performed from using flat start (mean and variances equal for all states)
    Training is performed using Baum Welch

    :param cluster: cluster number
    :param assignment: assignment of clusters for each sequence
    :param overlapping: the overlapping sequences
    :returns: a hidden markov model   
    """
    x_label = [overlapping[i] for i in range(0, len(overlapping)) if assignment[i] == cluster]
    frames  = int(np.mean([len(x) for x in x_label]))
    if frames > min_len:        
        logstructure.info("MkModel: {}".format("cluster"))
        logstructure.info("\t {} instances".format(len(x_label)))
        n = frames / 4
        l = 1 / n
        s = 1 - l

        trans_mat = DenseMarkovChain.from_probs([[s,   l/2, l/2, 0.0],
                                                 [0.0,   s, l,   0.0],
                                                 [0.0, 0.0, s,     l],
                                                 [0.0, 0.0, 0.0,   s]])

        trans_mat[Transition(START_STATE, 0)] = LogProb.from_float(1.0)
        trans_mat[Transition(3, STOP_STATE)]  = LogProb.from_float(1.0)

        dim       = len(overlapping[0][0])
        dists     = []
                    
        state = np.vstack(x_label)
        print("\t State: {}".format(state.shape))
        mu    = np.mean(state, axis=0)
        std   = np.std(state, axis=0) + 1e-4
        print("\t Stats: {} / {}".format(mu.shape, std.shape))
        dists = [Gaussian(mu, std) for i in range(0, 4)]

        logstructure.info("\t Model fit")
        hmm = HiddenMarkovModel(trans_mat, dists)
        for _ in range(0, max_train):
            logstructure.info("Cluster: {}\n{}".format(cluster, hmm.transitions))
            inference    = [infer.infer(hmm, seq) for seq in x_label]
            zetas        = [bw.infer(hmm, x_label[i], inference[i][1], inference[i][2]) for i in range(0, len(x_label))]    
            gammas       = [gamma for gamma, _, _ in inference]
            obs          = bw.continuous_obs(x_label, gammas, 1.0)
            diff         = np.sum([np.sum(np.square(a.mean - b.mean)) for (a, b) in zip(hmm.observations, obs)])
            transitions  = bw.markov(zetas, gammas)
            hmm.observations = obs
            hmm.transitions  = DenseMarkovChain.from_probs(np.exp(transitions))
            hmm.transitions[Transition(START_STATE, 0)] = LogProb.from_float(1.0)
            hmm.transitions[Transition(3, STOP_STATE)]  = LogProb.from_float(1.0)

            score = LogProb(ZERO)
            for gamma in gammas:
                for ll in gamma[-1]:
                    score = score + LogProb(ll)
            logstructure.info("Cluster: {} Diff: {}".format(score, diff))
        logstructure.info("Cluster: {}\n{}".format(cluster, hmm.transitions))
        return hmm
    return None


def decode(sequence, hmms):
    """
    Decode all sequences using a hidden Markov model
    :param sequence: a  sequences to decode
    :param hmms: a list of hidden markov model
    :returns: (max likelihoods, max assignment)
    """
    max_ll  = ZERO
    max_hmm = -1
    for i, hmm in enumerate(hmms):
        _, ll = viterbi(hmm, sequence)
        ll = ll.prob
        if ll > max_ll:
            max_ll = ll
            max_hmm = i
    return max_ll, max_hmm


def greedy_mixture_learning(sequences, hmms, th):
    """
    Greedily learn a mixture of hidden markov models [MIN].

    :param sequences: a list of sequences
    :param hmms: a list of hidden Markov models
    :param pool: a thread pool
    :param th: stop when improvement is below a threshold
    :returns: final set of hmms 
    """
    logstructure.info("Starting greedy mixture learning")
    last_ll = float('-inf')
    models   = []
    openlist = hmms.copy()
    while len(openlist) > 0:
        max_hypothesis_ll = float('-inf')
        max_hypothesis    = 0

        # find the model that when added to the hidden Markov models increases the likelihood most 
        # TODO Biggest Bottleneck
        for i, hmm in enumerate(openlist):
            hypothesis = models + [hmm]
            with mp.Pool(processes=10) as pool:
                decoded = pool.starmap(decode, ((sequence, hypothesis) for sequence in sequences))
            likelihoods = [ll for ll, c in decoded if c >= 0]
            likelihood  = sum(likelihoods) / len(likelihoods)
            if likelihood > max_hypothesis_ll:
                max_hypothesis_ll = likelihood
                max_hypothesis = i
        
        # assign the best model
        best   = openlist.pop(max_hypothesis)
        models = models + [best]
        # stop if adding the model did not change the likelihood 
        logstructure.info("Greedy Mixture Learning: {} {} {} {}".format(max_hypothesis_ll, len(openlist), len(models), max_hypothesis_ll - last_ll))
        if max_hypothesis_ll - last_ll < th:
            with mp.Pool(processes=10) as pool:
                decoded = pool.starmap(decode, ((sequence, models) for sequence in sequences))
            assignemnts = [assignment for _, assignment in decoded]
            return models, last_ll, assignemnts
        last_ll = max_hypothesis_ll
    with mp.Pool(processes=10) as pool:
        decoded = pool.starmap(decode, ((sequence, models) for sequence in sequences))
    assignemnts = [assignment for _, assignment in decoded]
    return models, last_ll, assignemnts


def hierarchical_clustering(
    annotation_path,
    max_dist = 2.5, 
    min_instances = 5,
    min_th= 8, 
    max_th= 500, 
    paa = 4, 
    sax = 5,
    processes = 10,
    max_instances=None
):
    """
    Hierarchical clustering of annotations
    :param annotation_path: path to work folder
    :param max_dist: distance threshold for clustering
    :param min_instances: minimum number of instances to start hmm
    :param min_len: minimum length of sequence
    :param max_len: maximum length of sequence
    :param paa: compressed size
    :param sax: quantization codebook size
    :param processes: number of threads
    :returns: clustering result [(start, stop, filename, cluster)]
    """
    logstructure.info('Max Instances: {}'.format(max_instances))
    # Find all overlapping sequences
    overlapping           = []
    for file in tf.io.gfile.listdir(annotation_path):        
        if file.startswith("embedding") and file.endswith(".csv"):
            path = "{}/{}".format(annotation_path, file)
            logstructure.info("\tReading {} {}".format(path, len(overlapping)))
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

    if max_instances is not None:
        overlapping = overlapping[:max_instances]
    overlapping = [x for x in overlapping if len(x[4]) > min_th and len(x[4]) < max_th]
    max_len = int(max([len(e) for _, _, _, _, e in overlapping]) + 1)
    sequences = [np.stack(s) for _, _, _, _, s in overlapping]

    logstructure.info("SAX Bucketing instances")
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
    
    logstructure.info("Bucketed Clustering")
    with mp.Pool(processes=processes) as pool:
        outputs = pool.starmap(process_dtw, ((assignment, overlapping, max_dist) for assignment, overlapping in by_assignment.items()))

    logstructure.info("Process Results")
    cur = 0
    assignments = []
    overlapping = []
    sequences   = []
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
                    sequences.append(sequence)
            for c in range(len(set(clustering))):
                cur += 1
    
    logstructure.info("Filter by cluster usage")
    sequences   = [sequences[i]   for i in range(len(sequences))   if n_instances[assignments[i]] > min_instances]
    overlapping = [overlapping[i] for i in range(len(overlapping)) if n_instances[assignments[i]] > min_instances]
    assignments = [assignments[i] for i in range(len(assignments)) if n_instances[assignments[i]] > min_instances]

    logstructure.info("Build Hidden Markov Models")
    model_pool = list(set(assignments))
    logstructure.info("Models: {}".format(len(model_pool)))
    with mp.Pool(processes=processes) as pool:
        hmms = pool.starmap(make_hmm, ((model, assignments, sequences) for model in model_pool))
        hmms = [hmm for hmm in hmms if hmm is not None]
        
    logstructure.info("Models: {}".format(len(hmms)))
    logstructure.info("Greedy Mixture Learning / Cluster Supression")
    models, last_ll, assignments = greedy_mixture_learning(sequences, hmms, 1e-4)
    pkl.dump(models, open('{}/hmms.pkl'.format(annotation_path), 'wb'))
    cluster_regions = [(start, stop, f, t, c) for c, (start, stop, f, t, _) in zip(assignments, overlapping) if c >= 0]
    return cluster_regions


def annotate_clustering(work_folder, annotations):
    """
    Annotates a clustering

    :param work_folder: folder with clustering results
    :param annotations: file with annotations
    :returns: dict[clusters][filename][start, stop, annotation]
    """
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
