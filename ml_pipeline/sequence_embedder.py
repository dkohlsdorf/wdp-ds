import numpy as np

from audio import *


class SequenceEmbedder:
    """
    Cut non silent spectrogram regions and embed them using our
    embedding model
    """

    def __init__(self, encoder, silence_detector, param):
        """
        :param encoder: a keras model In (?, T, D, 1) out (?, Latent)
        :param silence_detector: a keras model In(?, T, D, 1) out (?, 1)
        :param param: windowing parameters
        """
        self.encoder = encoder
        self.silence_detector = silence_detector
        self.param = param

    def embed(self, filename):
        """
        Embeds non silent regions from a file

        :param filename: path to file
        :param returns: iterator over (embedding, filename, start, stop)
        """
        batch = []
        regions = []
        for win in spectrogram_windows(filename, self.param):
            batch.append(win)
            if len(batch) == 1000:
                b = np.stack([x[0].reshape(x[0].shape[0], x[0].shape[1], 1) for x in batch]) 
                is_silence = self.silence_detector.predict(b)
                embedding  = self.encoder.predict(b)
                for i in range(0, len(batch)):
                    if int(round(is_silence[i][0])) == 0:
                        filename  = batch[i][1]
                        start     = batch[i][2]
                        stop      = batch[i][3]
                        print('- Found non silent region {} {}:{} extracting embedding of size {}'.format(filename, start / 48000, stop / 48000, embedding[i, :].shape))
                        regions.append((embedding[i, :], filename, start, stop))
                batch = []
        return regions
