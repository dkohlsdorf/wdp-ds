import random
import numpy as np
import pandas as pd
from numba import jit


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
    spectrogram = np.array(spectrogram)
    spectrogram = spectrogram[:, win//2:][:, lo:hi]
    return spectrogram


def resolve_window(sample_raw, window, step, fwd = True):
    if fwd:
        return int((sample_raw - window) / step)
    else:
        return int(sample_raw * step + window)

    
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
        

def dataset_unsupervised_regions_windowed(regions, wavfile, encoder, supervised, label_dict, lo, hi, win, step, T, l2_window, dont_window_whistle):
    df        = pd.read_csv(regions)
    N         = len(df)
    audio     = raw(wavfile) 
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
                x = encoder.predict(w, verbose=0)
                y = supervised.predict(w, verbose=0)

                n_wstl = 0
                n_others = 0
                for l in np.argmax(y, axis=1):
                    l = label_dict[l]
                    if l == 'WSTL_UP' or l == 'WSTL_DOWN':
                        n_wstl += 1
                    else:
                        n_others += 1
                print("FRAMES: {} / {}".format(n_wstl, n_others))
                if n_wstl > n_others and dont_window_whistle:
                    print(".. whole")
                    instances.append(x)
                    labels.append(y)
                    ids.append(i)
                else:
                    print(".. window")
                    # TODO add windowing info for export visuals
                    for j in range(l2_window, len(x), l2_window // 2):
                        instances.append(x[j - l2_window:j])
                        labels.append(y[j - l2_window:j])
                        ids.append(i)
    return ids, instances, labels

        
def dataset_unsupervised_regions(regions, wavfile, encoder, supervised, lo, hi, win, step, T):
    df        = pd.read_csv(regions)
    N         = len(df)
    audio     = raw(wavfile) 
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
                x = encoder.predict(w, verbose = 0)
                y = supervised.predict(w, verbose = 0)                
                instances.append(x)
                labels.append(y)
                ids.append(i)
    return ids, instances, labels


def dataset_unsupervised_windows(label, wavfile, lo, hi, win, step, raw_size, T, n = 10000):
    df = pd.read_csv(label)
    audio     = raw(wavfile)
    instances = []
    for _, row in df.iterrows():
        start = row['starts']
        stop  = row['stops']
        w     = audio[start:stop]
        if len(w) > 0:
            s = spectrogram(w, lo, hi, win, step)
            w = windowing(s, T)
            if w is not None:
                for i in range(0, len(w)):
                    instances.append(w[i])
    random.shuffle(instances)
    return instances[0:n]
    

def dataset_supervised_windows(label, wavfile, lo, hi, win, step, raw_size, label_dict = None):
    df        = pd.read_csv(label)
    df        = df.rename(columns=lambda x: x.strip())
    audio     = raw(wavfile)
    print("DIFF: {}".format(len(audio) - (df['offset'].max() + raw_size)))
    labels    = []
    instances = []
    ra = []
    
    fill = False
    if label_dict is None:
        label_dict = {}
        fill = True
        
    cur_label  = 0
    for _, row in df.iterrows():
        start = row['offset']
        stop  = start + raw_size
        label = row['annotation'].strip()
        if label not in label_dict and fill:
            label_dict[label] = cur_label
            cur_label += 1
        w = audio[start:stop]
        s = spectrogram(w, lo, hi, win, step)
        f, t = s.shape
        ra.append(w)
        instances.append(s)
        labels.append(label_dict[label])
    return instances, ra, labels, label_dict
