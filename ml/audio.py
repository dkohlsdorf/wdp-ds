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

def data_gen(paths, win, mk_lable = None):
    frame = 0
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.wav'):
                lable = None
                if mk_lable is not None:
                    lable = mk_lable(file)
                print('process file {} {}'.format(file, frame))
                fp = "{}{}".format(path, file)
                spec = spectrogram_from_file(fp) 
                (t, d) = spec.shape
                for i in range(win, t - 1, win // 2):
                    frame += 1
                    if lable is None:
                        x = spec[i - win:i]
                        mu  = np.mean(x)
                        std = np.std(x) + 1.0
                        x = (x - mu) / std
                        yield np.reshape(x, (win, 256, 1))
                    elif lable is 'predict_next':
                        x = spec[i - win:i + 1]
                        mu  = np.mean(x)
                        std = np.std(x) + 1.0
                        x = (x - mu) / std
                        y = x[-1, :]
                        x = x[:-1,:]
                        yield np.reshape(x, (win, 256, 1)), y
                    else:
                        x = spec[i - win:i]
                        x = np.reshape(x, (win, 256, 1))
                        mu  = np.mean(x)
                        std = np.std(x) + 1.0
                        yield ((x - mu) / std, lable)
