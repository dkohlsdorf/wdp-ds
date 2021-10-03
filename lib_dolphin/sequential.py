import numpy as np
from lib_dolphin.audio import *

MIN_LEN = 44100 // 10
MAX_LEN = 44100 // 2


def draw_noise(length, noise):
    N = len(noise)
    start  = np.random.randint(length, N - length)
    sample = noise[start: start + length]
    return sample


def draw_signal(ranges, signals, filtered_ids, filtered_predictions, filtered_instances, WIN = None):    
    i = np.random.randint(0, len(filtered_ids))
    start, stop = ranges[filtered_ids[i]]
    c = filtered_predictions[i]
    
    if WIN is not None and len(c) > WIN:
        for d in range(0, len(c[0])):
            c[:, d] = np.convolve(c[:, d], np.ones(WIN) / WIN, mode='same')

    labeling = []
    for j in range(len(c)):
        l = np.argmax(c[j])
        if l == 4:
            ci = 0
            labeling.append(0)
        else:
            ii = filtered_instances[i][j]
            ci = 1 + (l * 26 + clst[l].predict(ii.reshape(1, ii.shape[0]))[0])
            labeling.append(ci)
    return signals[start:stop], labeling


def combined(length, df, signals, noise, ranges, filtered_ids, filtered_predictions, filtered_instances, n = 10):
    noise = np.concatenate([draw_noise(length, noise) for i in range(n)])
    N = len(noise)        
    signal, c = draw_signal(ranges, signals, filtered_ids, filtered_predictions, filtered_instances)
    n = len(signal)
    
    if N-n > n:
        insert_at = np.random.randint(n, N-n)
    else:
        insert_at = 0
    p = np.random.uniform()
    noise_p = 1.0 - p
    w = int(min(n, N)) 
    if w == N:
        print("Cut: {} {}".format(n, N))        
    noise[insert_at:insert_at+w] = noise_p * noise[insert_at:insert_at+w] + p * signal[:w]
    return noise, insert_at, insert_at+w, c


def labels(start, stop, length, label):
    y = np.zeros((length))    
    start_fft = resolve_window(start, FFT_WIN, FFT_STEP)
    stop_fft  = resolve_window(stop, FFT_WIN, FFT_STEP) 
    w = T // 2    
    i         = resolve_window(len(label), T, T // 2, False)
    err       = stop_fft - start_fft - i
    binsz     = int(i / len(label))
    pos = start_fft
    for l in label:
        for i in range(binsz):
            if pos >= length:
                return y
            y[pos] = l
            pos += 1
    return y


def get_batch(signals, noise, df, filtered_instances, ranges, filtered_ids, filtered_predictions, n_clusters, fft_lo, fft_hi, win, step, batch = 1):
    batch_x = []
    batch_y = []
    y_discrete = []
    length = np.random.randint(MIN_LEN, MAX_LEN)
    for i in range(0, batch):
        x, start, stop, c = combined(length, df, signals, noise, ranges, filtered_ids, filtered_predictions, filtered_instances)
        spec              = spectrogram(x, fft_lo, fft_hi, win, step)
        
        N = length
        y = labels(start, stop, len(spec), c)
        y_hot = np.zeros((len(y), n_clusters))
        for i, l in enumerate(y): 
            l = int(l)     
            if l > n_clusters:
                print(l, y)
            y_hot[i, l] = 1.0        
        
        batch_x.append(spec.reshape((spec.shape[0], spec.shape[1], 1)))
        batch_y.append(y_hot)
        y_discrete.append(y)
    batch_x = np.stack(batch_x)
    batch_y = np.stack(batch_y)
    y_discrete = np.stack(y_discrete)
    return batch_x, batch_y, y_discrete
