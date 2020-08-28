import numpy as np

from audio import *
from collections import namedtuple
import logging
logging.basicConfig()
logembed = logging.getLogger('embedder')
logembed.setLevel(logging.INFO)


class SequenceEmbedder:
    """
    Cut non silent spectrogram regions and embed them using our
    embedding model
    """

    def __init__(self, encoder, param, silence_detector, type_classifier, clusterer):
        """
        :param encoder: a keras model In (?, T, D, 1) out (?, Latent)
        :param silence_detector: a keras model In(?, T, D, 1) out (?, 1)
        :param type_classifier: classifying spectrogram data into types
        :param param: windowing parameters
        """
        self.encoder = encoder
        self.silence_detector = silence_detector
        self.param = param
        self.type_classifier = type_classifier
        self.clusterer = clusterer


    def embed(self, filename, outpath, batch_sze=1000, th=15.0):
        """
        Embeds non silent regions from a file

        :param filename: path to file
        :param outpath: write csv file
        :param batch_sze: batch for embedding
        :param th: distance threshold
        """
        batch = []
        with open(outpath, "w") as fp:
            fp.write("filename\tstart\tstop\ttype\tcluster\tembedding\n")
            for win in spectrogram_windows(filename, self.param):
                batch.append(win)
                if len(batch) == batch_sze:
                    self.process_batch(batch, fp, th)
                    batch = []
            if len(batch) > 0:
                self.process_batch(batch, fp, th)
                batch = []

    def process_batch(self, batch, fp, th):
        b = np.stack([x[0].reshape(x[0].shape[0], x[0].shape[1], 1) for x in batch]) 
        is_silence = self.silence_detector.predict(b)
        types      = self.type_classifier.predict(b)                
        embedding  = self.encoder.predict(b)
        clustering = self.clusterer.transform(embedding)
        for i in range(0, len(batch)):
            if int(round(is_silence[i][0])) == 0:
                c = np.argmin(clustering[i])
                d = np.min(clustering[i])
                if d < th:
                    t         = np.argmax(types[i])                    
                    filename  = batch[i][1]
                    start     = batch[i][2]
                    stop      = batch[i][3]     
                    csv = ','.join(['%.5f' % f for f in embedding[i, :]])
                    fp.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(filename, start, stop, t, c, csv))
