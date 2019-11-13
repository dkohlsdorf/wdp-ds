from audio import * 

class SequenceEmbedder:
    '''
    Cut non silent spectrogram regions and embed them using our
    embedding model
    '''    

    def __init__(self, encoder, silence_detector, param):
        '''
        encoder: a keras model In (?, T, D, 1) out (?, Latent)
        silence_detector: a keras model In(?, T, D, 1) out (?, 1)
        param: windowing parameters
        '''
        self.encoder = encoder
        self.silence_detector = silence_detector
        self.param = param

    def embed(self, filename):
        '''
        Embeds non silent regions from a file
        
        filename: path to file
        returns: iterator over (embedding, filename, start, stop)
        '''
        for x in spectrogram_windows(filename, self.param):
            snippet = x[0].reshape(1, x[0].shape[0], x[0].shape[1], 1)
            is_silence = int(np.round(self.silence_detector.predict(snippet)[0]))
            if is_silence == 0:
                embedding = self.encoder.predict(snippet)
                filename  = x[1]
                start     = x[2]
                stop      = x[3]
                yield(embedding, filename, start, stop)
                