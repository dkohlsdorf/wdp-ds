import wave
from scipy.io import wavfile
import struct


class AudioSnippetCollection:
    '''
    Collection of audio snippets all in one file        
    '''

    def __init__(self, filename):
        '''
        :param filename: audio file to save in
        '''
        self.obj = wave.open(filename,'w')
        self.obj.setnchannels(1)
        self.obj.setsampwidth(2)
        self.obj.setframerate(48000)
        
    def write(self, data):
        '''
        write some audio followed by some zeros
        
        :param data: some data to attach
        '''
        for i in range(0, len(data)):
            b = struct.pack('<h', data[i])
            self.obj.writeframesraw(b)
        for i in range(48000 // 10):
            b = struct.pack('<h', 0)
            self.obj.writeframesraw(b)
    
    def close(self):
        '''
        close the file
        '''
        self.obj.close()
    
