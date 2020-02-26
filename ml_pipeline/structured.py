import numpy as np
import pandas as pd
import os 


from sklearn.cluster import AgglomerativeClustering
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
        for x, f, start, stop, t, c in regions:
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
        items = [item for _, _, _, item, _ in x]
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
    :returns: average distance
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


def braket_dist(x, y):
    '''
    Braketed distance for two sequences 
    compute the distance between their first elements and 
    last elements

    
    '''
    dtw = DTW(max([len(a) for a in x] + [len(b) for b in y]))
    dist_start = dtw(x[0],  y[0])
    dist_end   = dtw(x[-1], y[-1])
    return (dist_start + dist_end) / 2


def onion_peeler(annotation_path, min_group = 7, max_samples_appart=48000 * 30, max_dist = 5.0):
    '''
    Hierarchical clustering under braketing distance.
      - hierarchical clustering based on only the outer braket with distance threshold.
      - repeat for each cluster without the outer braket.
    
    TODO: Clustering by ID
    '''
    agg                   = AgglomerativeClustering(distance_threshold=max_dist,
                                                    metri='precomputed')
    header                = ["filename", "start", "stop", "type", "embedding"]
    df                    = pd.read_csv(annotation_path, sep="\t", header = None, names=header)
    signals               = df[df['type'] >= 2]
    signals['embedding']  = whistles['embedding'].apply(
        lambda x: np.array([float(i) for i in x.split(",")]))
    re                    = RegionExtractors(max_samples_appart)
    annotated             = [(row['start'], row['stop'], row['filename'], row['embedding'])
                             for _ , row in signals.iterrows()]
    overlapping           = groupBy(annotated, re.overlap)
    signal_groups         = groupBy(overlapping, re.close, min_group)
    for border in range(0, min_group // 2):
        dist = np.zeros(n, n)
        for i, (start_x, stop_x, f_x, embeddings_x) in enumerate(signal_groups):
            for j, (start_y, stop_y, f_y, embeddings_y) in enumerate(signal_groups):
                if i < j:
                    d = braket_dist(embedding_x[border:-border], embedding_y[border:-border])
                    dist[i, j] = d
                    dist[j, i] = d
        clustering = fit_predict = agg.fit(dist)
        
    
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
            
