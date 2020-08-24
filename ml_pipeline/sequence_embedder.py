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


    def embed(self, filename, batch_sze=1000, th=15.0):
        """
        Embeds non silent regions from a file

        :param filename: path to file
        :param returns: iterator over (embedding, filename, start, stop)
        """
        batch = []
        regions = []
        for win in spectrogram_windows(filename, self.param):
            batch.append(win)
            if len(batch) == batch_sze:
                b = np.stack([x[0].reshape(x[0].shape[0], x[0].shape[1], 1) for x in batch]) 
                is_silence = self.silence_detector.predict(b)
                types      = self.type_classifier.predict(b)                
                embedding  = self.encoder.predict(b)
                clustering = self.clusterer.transform(embedding)
                for i in range(0, len(batch)):
                    if int(round(is_silence[i][0])) == 0:
                        c = np.argmin(clustering[i])
                        d = np.max(clustering[i])
                        if d < th:
                            t         = np.argmax(types[i])                    
                            filename  = batch[i][1]
                            start     = batch[i][2]
                            stop      = batch[i][3]            
                            regions.append((filename, start, stop, t, c))
                batch = []
        return regions


class TypeExtraction(namedtuple("Induction", "clustering starts stops types files")):
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
            yield self.files[i], self.starts[i], self.stops[i], self.types[i], self.clustering[i]

    @classmethod
    def from_audiofile(cls, path, embedder):
        """
        Construct type annotations from folder with audio files

        :param folder: folder with audio files
        :param embedder: a sequence embedder
        :returns: type annotation
        """
        clustering = []
        starts     = []
        stops      = []
        types      = []
        files      = [] 
        logembed.info("- Working on embedding {}".format(path))
        regions = embedder.embed(path)
        logembed.info("\t- found region {}".format(len(regions)))
        for f, start, stop, t, c in regions:
            clustering.append(c)
            starts.append(start)
            stops.append(stop)
            types.append(t)
            files.append(f)
        return cls(clustering, starts, stops, types, files) 

    def save(self, path, append=False):
        mode = "w"
        if append:
            mode += "+"
        with open(path, mode) as fp:
            fp.write("filename\tstart\tstop\ttype\tcluster\n")
            for filename, start, stop, t, c in self.items():
                fp.write("{}\t{}\t{}\t{}\t{}\n".format(filename, start, stop, t, c))


