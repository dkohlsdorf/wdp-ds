import numpy as np
import pandas as pd
import os 

from collections import namedtuple
from dtw import DTW

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
    def from_audiofiles(cls, folder, embedder):
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
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                path = "{}/{}".format(folder, filename)
                print("- Working on embedding {}".format(path))
                regions = embedder.embed(path)
                for x, f, start, stop, t, c in regions:
                    embeddings.append(x)
                    starts.append(start)
                    stops.append(stop)
                    types.append(t)
                    files.append(f)
        return cls(embeddings, starts, stops, types, files) 

    def save(self, path):
        with open(path, "w") as fp:
            for filename, start, stop, t, embedding in self.items():
                csv = ','.join(['%.5f' % f for f in embedding])
                fp.write("{}\t{}\t{}\t{}\t{}\n".format(filename, start, stop, t, csv))


class RegionExtractors:

    def __init__(self, threshold = 0):
        self.threshold = threshold

    def overlap(self, x1, x2):
        '''
        Do two regions (x1_start, x1_stop, file) and (x2_start, x2_stop, file) overlap?

        :param x1: first region tuple 
        :param x2: second reggion tuple
        '''
        return max(x1[0],x2[0]) <= min(x1[1],x2[1]) and x1[2] == x2[2]

    def close(self, x1, x2):
        '''
        Are two regions (x1_start, x1_stop, file) and (x2_start, x2_stop, file) close in time?

        :param x1: first region tuple 
        :param x2: second reggion tuple
        '''
        return (x2[1] - x1[0]) < self.threshold and x1[2] == x2[2]


def mk_region(sequences):
    '''
    Convert a set of grouped sequences into a region

    :param sequences: [
        [(start, stop, file, x), (start, stop, file, x)], 
        [(start, stop, file, x)], 
        [(start, stop, file, x), (start, stop, file, x)]]
    :returns: [([x, x], start, stop), ([x], start, stop),([x, x], start, stop)]
    '''
    for x in sequences:
        items = [item for _, _, _, item in x]
        start = x[0][0]
        stop  = x[-1][1]
        f     = x[-1][2] 
        yield start, stop, f, items


def groupBy(sequences, grouping_cond, window_size=None):
    '''
    Extract groups of signals

    :param sequences: a sortd list of time stamped embedding sequences (seq, start, stop, file)
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


def interset_distance(x):
    '''
    Compute average distane in set of sequences

    :param x: numpy array (Instances, Time, Dimension)
    '''
    dtw = DTW(max([len(a) for a in x]))
    sum = 0
    n   = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dist, _ = dtw.align(x[i], x[j])
            sum += dist / (len(x[i]) * len(x[j]))
            n   += 1
    return sum / n


def signature_whistles(annotation_path, min_group = 3, max_samples_appart=48000 * 30, max_dist = 5.0):
    '''
    Extract signature whistles

    :param annotation_path: path to an annotation csv file
    :param min_group: minimum size of a group of whistles in order to be considered a signature whistle
    :param max_samples_appart: maximum number of samples between whistles to form a group
    :param max_dist: maximum distance allowed 
    '''
    print(annotation_path)
    header                = ["filename", "start", "stop", "type", "embedding"]
    df                    = pd.read_csv(annotation_path, sep="\t", header = None, names=header)
    whistles              = df[df['type'] == 3]
    whistles['embedding'] = whistles['embedding'].apply(lambda x: np.array([float(i) for i in x.split(",")]))
    re                    = RegionExtractors(max_samples_appart)
    annotated             = [(row['start'], row['stop'], row['filename'], row['embedding']) for _ , row in whistles.iterrows()]
    overlapping           = groupBy(annotated, re.overlap)
    signal_groups         = groupBy(overlapping, re.close, min_group)
    for start, stop, f, embeddings in signal_groups:
        embed_set  = [np.stack(e) for e in embeddings]
        inter_dist = interset_distance(embed_set)
        if inter_dist < max_dist:        
            yield start, stop, inter_dist, f


def signature_whistle_gaps(annotation_path, groups):
    '''
    Extract gaps from signature whistles

    :param annotation_path: path to an annotation csv file
    :param groups: sequence of (start, stop, distance, filename) representing signature whistles
    '''
    header = ["filename", "start", "stop", "type", "embedding"]
    df     = pd.read_csv(annotation_path, sep="\t", header = None, names=header)
    re     = RegionExtractors()
    gaps_whistle = []
    for start, stop, _, f in groups:
        signature_whistle = df[df["start"] >= start]
        signature_whistle = signature_whistle[signature_whistle["stop"] <= stop]
        signature_whistle = signature_whistle[signature_whistle["type"] == 3]
        gaps = []
        last = None
        for i, row in signature_whistle.iterrows():
            x = (row["start"], row["stop"], row["filename"])
            if last is not None:
                if not re.overlap(last, x):
                    gaps.append((last[1], x[0]))
            last = x
        gaps_whistle.append((gaps, f))
    return gaps_whistle
