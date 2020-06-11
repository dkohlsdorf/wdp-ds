import wave
import struct
import numpy as np

import audio
from scipy.io import wavfile


class AudioSnippetCollection:

    def __init__(self, filename):
        """
        :param filename: audio file to save in
        """
        params = audio.AudiofileParams.get()
        print(params)
        self.obj = wave.open(filename,'w')
        self.obj.setnchannels(1)
        self.obj.setsampwidth(params.sample_width)
        self.obj.setframerate(params.rate)
        
    def write(self, data):
        """
        write some audio followed by some zeros
        
        :param data: some data to attach
        """
        params = audio.AudiofileParams.get()
        frames = params.rate // 10
        b = bytearray(data)
        self.obj.writeframesraw(b)
        b = bytearray(np.zeros(frames))
        self.obj.writeframesraw(b)
    
    def close(self):
        """
        close the file
        """
        self.obj.close()
