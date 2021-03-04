import numpy as np
import pandas as pd


from numpy.fft        import fft
from scipy.io.wavfile import read, write    


def raw(path):
    x = read(path)[1]
    return x


def spectrogram(audio, lo = 20, hi = 200, win = 512, step=128):
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = np.abs(fft(audio[i - win: i] * hanning))
        spectrogram.append(dft)
    spectrogram = np.array(spectrogram)[:, win//2:][:, lo:hi]
    return spectrogram


def dataset_supervised(label, wavfile, whitelist):
    df    = pd.read_csv(label)
    audio = raw(wavfile)
    labels    = []
    instances = []
    windows   = []

    label_dict = {}
    cur_label  = 0
    for _, row in df.iterrows():
        start = row['offset']
        stop  = start + 5120
        label = row[' annotation'].strip()
        if label in whitelist:
            if label not in label_dict:
                label_dict[label] = cur_label
                cur_label += 1
            w = audio[start:stop]
            s = spectrogram(w)
            mu    = np.mean(s)
            sigma = np.std(s) + 1.0
            s     = (s - mu) / sigma
            f, t = s.shape
            instances.append(s)
            windows.append(w)
            labels.append(label_dict[label])
    return (windows, instances, labels, label_dict)
