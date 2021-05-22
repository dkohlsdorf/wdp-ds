import numpy as np
import pandas as pd


from numpy.fft        import fft
from scipy.io.wavfile import read, write    


def raw(path):
    try:
        x = read(path)[1]
        if len(x.shape) > 1:
            return x[:, 0]
        return x
    except:
        print("Could not read file: {}".format(path))
        return np.zeros(0)
    

def spectrogram(audio, lo = 20, hi = 200, win = 512, step=128, normalize=True):
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = np.abs(fft(audio[i - win: i] * hanning))
        if normalize:
            mu  = np.mean(dft)
            std = np.std(dft) + 1.0
            spectrogram.append((dft - mu) / std)
        else:
            spectrogram.append(dft)        
    spectrogram = np.array(spectrogram)[:, win//2:][:, lo:hi]
    return spectrogram


def dataset_supervised(label, wavfile, lo = 20, hi = 200, win = 512, step=128, raw_size=5120):
    df    = pd.read_csv(label)
    audio = raw(wavfile)
    labels    = []
    instances = []
    windows   = []

    label_dict = {}
    cur_label  = 0
    for _, row in df.iterrows():
        start = row['offset']
        stop  = start + raw_size
        label = row[' annotation'].strip()
        if label not in label_dict:
            label_dict[label] = cur_label
            cur_label += 1
        w = audio[start:stop]
        s = spectrogram(w, lo, hi, win, step)
        f, t = s.shape
        instances.append(s)
        windows.append(w)
        labels.append(label_dict[label])
    return (windows, instances, labels, label_dict)