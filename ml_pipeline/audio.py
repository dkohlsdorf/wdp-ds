import numpy as np

import matplotlib.pyplot as plt

from numpy.fft import fft
from sksound.sounds import Sound
from collections import namedtuple

class WindowParams(namedtuple('WindowParams', 'spec_win spec_step fft_win fft_step')):

    @property
    def win_len(self):
        return self.spec_win * (self.fft_win + self.spec_step)

    @property
    def step(self):
        return self.spec_step * self.fft_step 
    
    def range(self, sample):
        '''
        The range of a sample in a spectrogam window in the audio file

        sample: a sample in the spectrogram window
        '''        
        start_audio = sample * self.step
        stop_audio  = start_audio + self.win_len
        print(">>>", self.win_len)
        return start_audio, stop_audio

    def len(self, audio_samples):
        '''
        Compute the number of spectrogram window in an audio file

        audio_samples: number of audio samples in audio_file
        '''
        return (audio_samples - self.win_len) // self.step + 1

        
def spectrogram_windows(filename, params):
    assert isinstance(params, WindowParams)
    sound = Sound(filename)        
    data  = sound.data        
    if len(sound.data.shape) > 1:
        data = np.mean(data, axis=1) 
    n = len(data)
    n_windows = params.len(n)

    for i in range(0, n_windows):
        start, stop = params.range(i)
        audio = data[start:stop]
        spec = fwd_spectrogram(audio, params.fft_win, params.fft_step)
        print(spec.shape)

def fwd_spectrogram(audio, win=512, step=64):
    '''
    Compute the spectrogram of audio data

    audio: one channel audio
    win: window size for dft sliding window
    step: step size for dft sliding windo
    '''
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        start = win // 2        
        dft = np.abs(fft(audio[i - win: i] * hanning))[start:win]
        spectrogram.append(dft)
    return np.array(spectrogram)

spectrogram_windows('data/demo/06111101.wav', WindowParams(128, 64, 512, 64))
