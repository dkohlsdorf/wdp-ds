import struct
import numpy as np
import pandas as pd
from numba import jit


from numpy.fft        import fft
from scipy.io.wavfile import read, write    


PERIOD      = 1
SAMPLE_SIZE = 4 
USER        = 9
ENDIEN      = 'big'


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


@jit
def windowing(region, window):
    N, D = region.shape
    windows = []
    if N > window:
        step = window // 2
        for i in range(window, N, step):
            r = region[i - window:i].reshape((window, D, 1))
            windows.append(r)
        return np.stack(windows)
    else:
        return None
        

def dataset_unsupervised_regions(regions, wavfile, encoder, supervised, lo, hi, win, step, T):
    df        = pd.read_csv(regions)
    N         = len(df)
    instances = []
    labels    = []
    ids       = []
    for i, row in df.iterrows():
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            w = windowing(s, T)
            if w is not None:
                if i % 100 == 0:
                    print(" ... reading {}/{}={}".format(i, N, i / N))
                x = encoder.predict(w)
                y = supervised.predict(w)                
                instances.append(x)
                labels.append(y)
                ids.append(i)
    return ids, instances, labels


def dataset_supervised_windows(label, wavfile, lo, hi, win, step, raw_size):
    df        = pd.read_csv(label)
    audio     = raw(wavfile)
    labels    = []
    instances = []

    label_dict = {}
    cur_label  = 0
    for _, row in df.iterrows():
        start = row['offset']
        stop  = start + raw_size
        label = row['annotation'].strip()
        if label not in label_dict:
            label_dict[label] = cur_label
            cur_label += 1
        w = audio[start:stop]
        s = spectrogram(w, lo, hi, win, step)
        f, t = s.shape
        instances.append(s)
        labels.append(label_dict[label])
    return instances, labels, label_dict


def write_htk(sequence, to):
    n = len(sequence)
    dim = len(sequence[0])
    with open(to, "wb") as f:
        sze = dim * SAMPLE_SIZE
        f.write(n.to_bytes(4, byteorder=ENDIEN))
        f.write(PERIOD.to_bytes(4, byteorder=ENDIEN))
        f.write(sze.to_bytes(2, byteorder=ENDIEN))
        f.write(USER.to_bytes(2, byteorder=ENDIEN))
        for i in range(0, n):
            for j in range(0, dim):
                x = sequence[i, j]
                ba = bytearray(struct.pack(">f", x))
                f.write(ba)
                
