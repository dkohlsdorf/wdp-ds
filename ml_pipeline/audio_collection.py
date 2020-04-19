import wave
import struct
import numpy as np

from scipy.io import wavfile


class AudioSnippetCollection:
    '''
    Collection of audio snippets all in one file        
    '''

    def __init__(self, filename, sample_width=4):
        '''
        :param filename: audio file to save in
        '''
        self.obj = wave.open(filename,'w')
        self.obj.setnchannels(1)
        self.obj.setsampwidth(sample_width)
        self.obj.setframerate(48000)
        
    def write(self, data, frames = 48000 // 10):
        '''
        write some audio followed by some zeros
        
        :param data: some data to attach
        '''
        b = bytearray(data.astype(np.int32))
        self.obj.writeframesraw(b)
        b = bytearray(np.zeros(frames, dtype=np.int32))
        self.obj.writeframesraw(b)
    
    def close(self):
        '''
        close the file
        '''
        self.obj.close()
