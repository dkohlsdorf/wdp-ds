import numpy as np
import random

from numpy.fft import fft
from scipy.io import wavfile
from collections import namedtuple


class WindowParams(namedtuple('WindowParams', 'spec_win spec_step fft_win fft_step highpass')):

    @property
    def fft_win_filtered(self):
        '''
        Pad the FFT window by the highpass so the target dimension stays the same
        '''
        return self.fft_win + 2 * self.highpass
    
    @property
    def win_len(self):
        '''
        Length of the spectrogram window returned in audio samples
        '''
        return self.spec_win * self.fft_step + self.fft_win_filtered

    @property
    def step(self):
        '''
        Step size of spectrogram window in audio samples
        '''
        return self.spec_step * self.fft_step 
    
    def range(self, sample):
        '''
        The range of a sample in a spectrogam window in the audio file

        sample: a sample in the spectrogram window

        returns: start sample in raw audio, stop sample in raw audio
        '''        
        start_audio = sample * self.step 
        stop_audio  = start_audio + self.win_len
        return start_audio, stop_audio

    def len(self, audio_samples):
        '''
        Compute the number of spectrogram window in an audio file

        audio_samples: number of audio samples in audio_file

        return: length of the window
        '''
        return (audio_samples - self.win_len) // self.step + 1


def labeled_spectrogram_windows(filename, params, shuffle=False, label_func):
    '''
    Generate spectrogram windows from file as well as labels
    generated from the filename or spectrogram.
    
    For example:
      a binary classifier: f(name, x) => 1 if name == 'noise' else 0
      an auto encoder:     f(name, x) => x
    filename: the filename
    params: parameters 
    shuffle: shuffle the dataset
    label_func: f(filename, spectrogram) => target 
    
    returns: (spectrogram, label, filename, start, stop)
    '''
    for (spectrogram, _, start, stop) in spectrogram_windows(filename, params, shuffle):
        label = label_func(filename, spectrogram)
        yield (spectrogram, label, filename, start, stop)

        
def spectrogram_windows(filename, params, shuffle=False):
    '''
    Extract all spectrogram windows from an audio file.
    Also z-normalizes the spectrograms

    params: Windowing parameters
    highpass: Frequency below which we cut the spectrogram
    
    returns: (spectrogram, filename, start, stop)
    '''    
    assert isinstance(params, WindowParams)
    _, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1) 
    n = len(data)
    n_windows = params.len(n)
    ordered   = [i for i in range(0, n_windows)]
    if shuffle:
        random.shuffle(ordered)
    for i in ordered:
        start, stop = params.range(i)
        audio = data[start:stop]
        spec  = fwd_spectrogram(audio, params.fft_win_filtered, params.fft_step)
        dft_start = params.fft_win - params.fft_win//2
        dft_stop  = params.fft_win 
        spec  = spec[:, dft_start:dft_stop]
        mu      = np.mean(spec)
        sigma   = np.std(spec) + 1.0
        yield ((spec - mu) / sigma, filename, start, stop)

        
def fwd_spectrogram(audio, win=512, step=64):
    '''
    Compute the spectrogram of audio data

    audio: one channel audio
    win: window size for dft sliding window
    step: step size for dft sliding windo

    return: power spectrum
    '''
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        start = win // 2                
        dft = np.abs(fft(audio[i - win: i] * hanning))
        spectrogram.append(dft)
    return np.array(spectrogram)
