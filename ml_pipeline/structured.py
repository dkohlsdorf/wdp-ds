import numpy as np
import pandas as pd
import os 

from collections import namedtuple


class TypeExtraction(namedtuple("Induction", "embeddings starts stops types files")):
    """
    Type annotations for dolphin communication
    """

    @property
    def len(self):
        return len(starts)

    def items(self):        
        """
        Iterate annotated tuples (filename, start, stop, type, embedding vector)
        """
        n = self.len()
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

    def __init__(self, threshold):
        self.threshold = threshold

    def overlap(x1, x2):
        '''
        Do two regions (x1_start, x1_stop, file) and (x2_start, x2_stop, file) overlap?

        :param x1: first region tuple 
        :param x2: second reggion tuple
        '''
        return max(x1[0],x2[0]) <= min(x1[1],x2[1]) and x1[2] == x2[2]

    def close(x1, x2):
        '''
        Are two regions (x1_start, x1_stop, file) and (x2_start, x2_stop, file) close in time?

        :param x1: first region tuple 
        :param x2: second reggion tuple
        '''
        (x2[2] - x1[1]) < self.threshold and x1[3] == x2[3]


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
        yield items, start, stop, f


def groupBy(sequences, grouping_cond):
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
            else:
                groups.append(current)
                current = [x]
    return mk_region(groups)


def interset_distance(x):
    '''
    Compute average distane in set of sequences

    :param x: numpy array (Instances, Time, Dimension)
    '''
    sum = 0
    n   = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sum += dtw(x[i], x[j]) / (len(x[i]) * len(x[j]))
            n   += 1
    return sum / n


def signature_whistles(annotation_path, min_group = 3, max_samples_appart=48000 * 15, max_dist = 10.0):
    '''
    Extract signature whistles

    :param annotation_path: path to an annotation csv file
    :param min_group: minimum size of a group of whistles in order to be considered a signature whistle
    :param max_samples_appart: maximum number of samples between whistles to form a group
    :param max_dist: maximum distance allowed 
    '''
    header                = ["filname", "start", "stop", "type", "embedding"]
    df                    = pd.read_csv(annotation_path, sep="\t", header = None, names=header)
    whistles              = df[df['type'] == 3]
    whistles['embedding'] = whistles['embedding'].apply(lambda x: np.array([float(i) for i in x.split(",")]))
    re                    = RegionExtractors(max_samples_appart)
    annotated             = [(row['start'], row['stop'], row['file'], row['embedding']) for _ , row in annotation_df.iterrows()]
    overlapping           = groupBy(annotated, re.overlap)
    signal_groups         = groupBy(overlapping, re.close)
    for embeddings, start, stop, f in signal_groups:
        embed_set  = np.stack(embeddings)
        inter_dist = interset_distance(embed_set)
        if len(embeddings) >= min_group and inter_dist < max_dist:        
           yield start, stop, f



    


