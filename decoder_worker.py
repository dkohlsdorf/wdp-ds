import pickle as pkl
import sys
import time
import heapq

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple
from tensorflow.keras.models import load_model

from lib_dolphin.audio import *
from lib_dolphin.sequential import *
from lib_dolphin.eval import *
from lib_dolphin.discrete import *


SPLIT_SEC    = 60
SPLIT_RATE   = 44100
SPLIT_SKIP   = 0.5

FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 230
FFT_LO       = 100

D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)


NEURAL_NOISE_DAMPENING = 1.0
NEURAL_SMOOTH_WIN      = 64
NEURAL_SIZE_TH         = 32


def split(audio_file):
    window_size = SPLIT_SEC * SPLIT_RATE
    skip        = int(window_size * SPLIT_SKIP)
    x           = raw(audio_file)
    n           = len(x)

    regions = []
    for i in range(window_size, n, skip):
        regions.append(x[i-window_size:i])
    return regions


def spec(x):
    return spectrogram(x, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)


def decode(x, decoder, label_mapping):
    t, d = x.shape
    print(x.shape)
    a = x.reshape((1,t,d,1))
    p = decoder.predict(a).reshape((a.shape[1], label_mapping.n + 1)) 
    if len(p) > NEURAL_SMOOTH_WIN:
        for i in range(0, len(p[0])):
            p[:, i] = np.convolve(p[:, i], np.ones(NEURAL_SMOOTH_WIN) / NEURAL_SMOOTH_WIN, mode='same')
    p[:, 0] *= NEURAL_NOISE_DAMPENING
    local_c = p.argmax(axis=1)
    return local_c

    
def ngrams(sequence, n=12):
    results = []
    for i in range(n, len(sequence)):
        x = [s.cls for s in sequence[i-n:i]]
        x = " ".join(x)
        results.append(x)
    return results


def match(sequence, db, n=12):
    ids = []
    for k in ngrams(sequence):
        if k in db:
            ids.extend(db[k])
    return set(ids)


def query(sequence, db, sequences):
    ids = match(sequence, db)
    pq  = [] 
    for i in ids:
        d = levenstein(sequence, sequences[i])
        heapq.heappush(pq, (d, i))
    return [heapq.heappop(pq) for _ in ids]
    

if __name__ == '__main__':
    print("Decoder")    
    decoder  = load_model('../results/decoder_nn.h5')
    lab      = pkl.load(open("../results/labels.pkl", "rb"))
    reverse  = {v:k for k, v in lab.items()}
    label_mapping = pkl.load(open('../results/label_mapping.pkl', 'rb'))

    start = time.time()
    filename = "../data/dolphin.wav"
    x = split(filename)
    db = {}
    sequences = []
    for i in range(len(x)):
        s    = spec(x[i])
        dec  = decode(s, decoder, label_mapping)
        c    = compress_neural(dec, len(s), reverse, label_mapping)
        plot_neural(s, c, f"spec_{i}.png")
        keys = ngrams(c)
        for k in keys:
            if k not in db:
                db[k] = []
            db[k].append(i)
        sequences.append(c)
        if i % 10 == 0 and i > 0:
            stop = time.time()
            secs = stop - start
            print("Execute 10 minutes {} [seconds]".format(int(secs)))
            start = time.time()
    
    print(len(sequences), query(sequences[0], db, sequences))