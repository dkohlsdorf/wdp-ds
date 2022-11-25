import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lib_dolphin.audio import *
from collections import namedtuple
from lib_dolphin.eval import *
from lib_dolphin.parameters import *


def raven(path, symbols, sep='\t'):    
    headers = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
               "Delta Time (s)", "Delta Freq (Hz)", "Avg Power Density (dB FS/Hz)", "Annotation"]
    header = sep.join(headers)
    table = [header]
    for i, symbol in enumerate(symbols):     
        if symbol.cls[0] != '_':
            x_start    = symbol.start * FFT_STEP
            x_end      = symbol.stop * FFT_STEP
            secs_start = x_start / 44100
            secs_end   = x_end / 44100
            f_start    = 0
            f_end      = 44100 // 2
            channel    = 1
            annotation = symbol.cls
            dt         = ""
            df         = ""
            power      = ""
            table.append(f"{i+1}{sep}Waveform 1{sep}1{sep}{secs_start}{sep}{secs_end}{sep}{f_start}{sep}{f_end}{sep}{dt}{sep}{df}{sep}{power}{sep}{annotation}")
            table.append(f"{i+1}{sep}Spectrogram 1{sep}1{sep}{secs_start}{sep}{secs_end}{sep}{f_start}{sep}{f_end}{sep}{dt}{sep}{df}{sep}{power}{sep}{annotation}")
    with open(path, "w") as fp:
        fp.write("\n".join(table))


def split(audio_file):
    window_size = SPLIT_SEC * SPLIT_RATE
    skip        = int(window_size * SPLIT_SKIP)
    x           = raw(audio_file)
    n           = len(x)

    regions = []
    bounds  = []
    if n < window_size:
        return [x], [(0, n)], audio_file
    
    for i in range(window_size, n, skip):
        regions.append(x[i-window_size:i])
        bounds.append((i-window_size, i))
    return regions, bounds, audio_file


class DecodedSymbol(namedtuple('DecodedSymbol', 'start stop cls id')):
    
    @classmethod
    def from_dict(cls, x):
        return cls(x['start'], x['stop'], x['cls'], x['id'])
    
    def to_dict(self):
        return {
            "cls":   self.cls,                
            "start": self.start,
            "stop":  self.stop,
            "id":    self.id                        
        }
    

def compress_neural(decoding, n, reverse, label_mapping):
    classifications = []
    last = 0
    for i in range(1, len(decoding)):
        if decoding[i] != decoding[i - 1]:                                   
                
            start = last
            stop = i
            leng = stop - start
            if decoding[i - 1] == 0:  
                decoding[i - 1] = sil_label(leng)
                
            if leng > NEURAL_SIZE_TH:
                d = DecodedSymbol(start, stop,  i2name(decoding[i - 1], reverse, label_mapping), decoding[i - 1])
                classifications.append(d)
            last = i
    if last != n:
        start = last
        stop  = n
        leng  = stop - start        
        if decoding[-1] == 0:  
            decoding[-1] = sil_label(leng)
        d = DecodedSymbol(last, n,  i2name(decoding[-1], reverse, label_mapping), decoding[-1])        
        classifications.append(d)                                
    return classifications
    
    
def plot_neural(spectrogram, compressed, img_path):    
    fig, ax = plt.subplots()
    fig.set_size_inches(len(spectrogram) / 100, len(spectrogram[0]) / 100)
    ax.imshow(BIAS - spectrogram.T * SCALER, cmap='gray')                  
    for region in compressed:        
        if region.id > 0:
            rect = patches.Rectangle((region.start, 0), region.stop - region.start,
                             256, linewidth=1, edgecolor='r', facecolor=COLORS[region.id])
            ax.add_patch(rect)
            plt.text(region.start + (region.stop - region.start) // 2 , 30, region.cls, size=12)
    if img_path is not None:
        plt.savefig(img_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

    
def reject(x, p, th):
    if p < th:
        return 0
    else:
        return x


def i2name(i, reverse, label_mapping):    
    if i == 0:
        return '_'
    elif i == -1:
        return '__'
    elif i == -2:
        return '___'
    else:
        c, n = label_mapping.bwd(i)
        l = reverse[c]        
        if "DOWN" in l:
            l = 'D'
        elif "UP" in l:
            l = 'U'
        else:
            l = l[0]
            
        return f'{l}{chr(97 + (n - 1))}'

    
def sil_label(leng):
    if leng < 35:
        return 0
    if leng < 80:
        return -1
    return -2



class LabelMapping(namedtuple('LabelMapping', 'prefix')):
    
    @classmethod
    def mapping(cls, clusters):
        N = len(clusters)
        sizes = np.zeros(N + 1, dtype=np.int32)
        for i, kmeans in clusters.items():
            sizes[i + 1] = kmeans.n_clusters
        prefixes = np.cumsum(sizes)    
        return LabelMapping(prefixes)
    
    def fwd(self, label, cluster):
        return self.prefix[label] + cluster
    
    def bwd(self, i):
        j = 0
        while j < len(self.prefix) and i > self.prefix[j]:
            j += 1
        j = j - 1
        c = i - self.prefix[j]
        return j, c
    
    @property
    def n(self):
        return self.prefix[-1]

    
def draw_noise(length, noise):
    N = len(noise)
    start  = np.random.randint(length, N - length)
    sample = noise[start: start + length]
    return sample


def draw_signal(ranges, signals, ids, predictions, instances, clst, label_mapping, WIN = None, verbose=False, no_noise=True):    
    i = np.random.randint(0, len(ids))
    start, stop = ranges[ids[i]]
    c = predictions[i]
    
    if WIN is not None and len(c) > WIN:
        for d in range(0, len(c[0])):
            c[:, d] = np.convolve(c[:, d], np.ones(WIN) / WIN, mode='same')

    labeling = []
    total_score = 0
    n_nonzero = 0
    for j in range(len(c)):
        if no_noise:
            c[j][4] = -1.0
        l = np.argmax(c[j])
        if l == 4:
            ci = 0
            labeling.append(0)
        else:
            ii = instances[i][j]
            cluster_number = clst[l].predict(ii.reshape(1, ii.shape[0]))[0]
            score = clst[l].score(ii.reshape(1, ii.shape[0]))
            ci = 1 + label_mapping.fwd(l, cluster_number)            
            labeling.append(ci)
            total_score += score
            n_nonzero   += 1
    total_score /= (1 + n_nonzero)
    if verbose:
        print(f" ... kmeans fit: {total_score}")            
    return signals[start:stop], labeling, total_score, n_nonzero


def combined(length, signals, noise, ranges, ids, predictions, instances, clst, label_mapping, n = 10, min_signal=0.75, score_threshold=None, n_trials = 250, verbose=False):
    
    noise = np.concatenate([draw_noise(length, noise) for i in range(n)])
    N = len(noise)    
    
    trials = 0    
    signal, c, score, n_nonzero = draw_signal(ranges, signals, ids, predictions, instances, clst, label_mapping)
    while score_threshold is not None and score < score_threshold and trials < n_trials and n_nonzero > 0:
        signal, c, score, n_nonzero= draw_signal(ranges, signals, ids, predictions, 
                                       instances, clst, label_mapping)
        trials += 1      
        
    n = len(signal)    
    if N-n > n:
        insert_at = np.random.randint(n, N-n)
    else:
        insert_at = 0
    p = min(min_signal, np.random.uniform())
    noise_p = 1.0 - p
    w = int(min(n, N)) 
    if w == N and verbose:
        print("Cut: {} {}".format(n, N))        
    if verbose:
        print(f"Noise to Signal Ratio: noise = {N} signal = {n_nonzero}/{n} ||||| {n/N * 100}%")
    noise[insert_at:insert_at+w] = noise_p * noise[insert_at:insert_at+w] + p * signal[:w]
    return noise, insert_at, insert_at+w, c


def labels(start, stop, length, label, fft_win, fft_step, T):
    y = np.zeros((length))    
    start_fft = resolve_window(start, fft_win, fft_step)
    stop_fft  = resolve_window(stop, fft_win, fft_step)
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


def get_batch(signals, noise, instances, ranges, ids, predictions, n_clusters, clst, label_mapping, fft_lo, fft_hi, win, step, T, batch = 1):
    batch_x = []
    batch_y = []
    y_discrete = []
    length = np.random.randint(MIN_LEN, MAX_LEN)
    for i in range(0, batch):
        x, start, stop, c = combined(length, signals, noise, ranges, ids, predictions, instances, clst, label_mapping)
        spec              = spectrogram(x, fft_lo, fft_hi, win, step)
        
        N = length
        y = labels(start, stop, len(spec), c, win, step, T)
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
