import pickle as pkl
import sys
import time
import heapq

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import polling

from collections import namedtuple
from tensorflow.keras.models import load_model

from lib_dolphin.audio import *
from lib_dolphin.sequential import *
from lib_dolphin.eval import *
from lib_dolphin.discrete import *

from redis import Redis
from datetime import datetime 



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


def match(sequence, db, n):
    ids = []
    for k in ngrams(sequence):
        if k in db:
            ids.extend(db[k])
    return list(set(ids))


def labels(s):
    return [c.cls for c in s]


def query(sequence, db, sequences, n = 4):
    ids = match(sequence, db, n)
    pq  = [] 
    for i in ids:
        d = levenstein(labels(sequence), labels(sequences[i]))
        heapq.heappush(pq, (d, i))
    return [heapq.heappop(pq) for _ in ids]
    
    
def knn(sequence, sequences, ids, k):
    pq = []
    for i in range(0, min(k, len(ids))):
        d = levenstein(labels(sequence), labels(sequences[ids[i]]))
        heapq.heappush(pq, (-d, ids[i]))
    if k < len(ids):
        for i in range(k, len(ids)):
            d = levenstein(labels(sequence), labels(sequences[ids[i]]))
            if d < -pq[0][0]:
                heapq.heappush(pq, (-d, ids[i]))
                heapq.heappop(pq)
    result = reversed([heapq.heappop(pq) for i in range(0, len(pq))])
    return [(-1 * d, i) for d, i in result]
    
    
def discovery(sequences, db, k=2, n=2):
    neighbors = {}
    densities = {}
    for i, sequence in enumerate(sequences):
        ids = match(sequence, db, n)
        nn  = knn(sequence, sequences, ids, k)
        neighbors[i] = nn
        if len(nn) == k:
            densities[i] = 1. / nn[-1][0]
    return densities, neighbors


class DecodingWorker:
    
    KEY = 'WDP-DS'
    
    def __init__(self, model_path, image_path, redis):
        self.decoder       = load_model(f'{model_path}/decoder_nn.h5')
        self.lab           = pkl.load(open(f"{model_path}/labels.pkl", "rb"))
        self.reverse       = {v:k for k, v in self.lab.items()}
        self.label_mapping = pkl.load(open(f'{model_path}/label_mapping.pkl', 'rb'))
        self.image_path    = image_path
        self.db            = {}
        self.sequences     = []
        self.redis         = redis
        
    def work(self):
        now = datetime.now()        
        result = self.redis.lpop(DecodingWorker.KEY)
        print(f'.. Check for work {now} {result}')
        if result is not None:
            filename = result   
            print(f'.. Work: {filename}')
            x = split(filename)
            start = time.time()        
            for i in range(len(x)):
                s    = spec(x[i])
                dec  = decode(s, self.decoder, self.label_mapping)
                c    = compress_neural(dec, len(s), self.reverse, self.label_mapping)
                plot_neural(s, c, f"{self.image_path}/spec_{i}.png")
                keys = ngrams(c)
                for k in keys:
                    if k not in self.db:
                        self.db[k] = []
                    self.db[k].append(i)
                self.sequences.append(c)

                if i % 10 == 0 and i > 0:
                    stop = time.time()
                    secs = stop - start
                    print("Execute 10 minutes {} [seconds]".format(int(secs)))
                    start = time.time()

                    
        
if __name__ == '__main__':

    '''
    TODO: save and load DecodingWorker
    TODO: rest service
    '''
    
    if sys.argv[1] == 'worker':
        print("Decoding Worker")    
        worker = DecodingWorker('../web_service/ml_models/', '../web_service/images/', Redis())
        polling.poll(lambda: worker.work(), step=5, poll_forever=True)        
    elif sys.argv[1] == 'enqueue':
        print('Batch Enqueue')
        folder = sys.argv[2]
        r = Redis()
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                path = f'{folder}/{filename}'
                print(f" .. Enqueue: {path}")
                r.lpush(DecodingWorker.KEY, path)
