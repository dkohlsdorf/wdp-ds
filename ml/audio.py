import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.io import wavfile


def fwd_spectrogram(audio, win=512, step=256):
    '''
    Compute the spectrogram of audio data

    audio: one channel audio
    win: window size for dft sliding window
    step: step size for dft sliding windo
    '''
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = fft(audio[i - win: i] * hanning)
        spectrogram.append(dft)
    return np.array(spectrogram)


def spectrogram_from_file(filename, win=512, step=256):
    '''
    Read audio and convert to z-normalized spectrogram  
    filename: path to the file
    max_len: clip files
    '''
    _, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]    
    start = win // 2
    spec = np.abs(fwd_spectrogram(data))[:, start:win]
    return spec

