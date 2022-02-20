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

from fastavro import writer, reader, parse_schema


SCHEMA = {
    "name": "WDP_Decoded",
    "namespace": "wdp",
    "type": "record",
    "fields": [
        {"name": "path",     "type": "string"},
        {"name": "start",    "type": "int"},
        {"name": "stop",     "type": "int"},
        {"name": "sequence", "type": {
            "type": "array", 
            "items": {
                "name": "tokens",
                "type": "record", 
                "fields": [
                    {"name": "cls",      "type": "string"},                
                    {"name": "start",    "type": "int"},
                    {"name": "stop",     "type": "int"},
                    {"name": "id",       "type": "int"}                
                ]}
            }        
        }
    ]
}


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
    bounds  = []
    for i in range(window_size, n, skip):
        regions.append(x[i-window_size:i])
        bounds.append((i-window_size, i))
    return regions, bounds, audio_file


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
    
    def __init__(self, model_path, image_path, sequence_path, redis):
        self.decoder       = load_model(f'{model_path}/decoder_nn.h5')
        self.lab           = pkl.load(open(f"{model_path}/labels.pkl", "rb"))
        self.reverse       = {v:k for k, v in self.lab.items()}
        self.label_mapping = pkl.load(open(f'{model_path}/label_mapping.pkl', 'rb'))
        self.image_path    = image_path
        self.sequence_path = sequence_path
        
        self.sequences     = []        
        self.redis         = redis
        self.schema        = parse_schema(SCHEMA)
        
    def discovery(self):
        db = {}
        for filename, start, stop, sequence in self.sequences:
            keys = ngrams(sequence)
            for k in keys:
                if k not in self.db:
                    db[k] = []
                db[k].append([filename, start, stop])            
        densities, neighbors = discovery(self.sequences, db)
        
    def work(self):
        now = datetime.now()        
        result = self.redis.lpop(DecodingWorker.KEY)
        
        print(f'.. Check for work {now} {result}')        
        if result is not None:
            records = []
            
            filename = result
            file_id = str(filename).split('/')[-1].split('.')[0]             
            print(f'.. Work: {filename} {file_id}')
            regions, bounds, audio_file = split(filename)
            start = time.time()        
            for i in range(len(regions)):
                s                       = spec(regions[i])
                start_bound, stop_bound = bounds[i] 
                dec  = decode(s, self.decoder, self.label_mapping)
                c    = compress_neural(dec, len(s), self.reverse, self.label_mapping)
                plot_neural(s, c, f"{self.image_path}/{file_id}_{start_bound}_{stop_bound}.png")
                self.sequences.append((filename, start_bound, stop_bound, c))
                
                records.append({                
                    "path":     str(filename),
                    "start":    start_bound,
                    "stop":     stop_bound,
                    "sequence": [token.to_dict() for token in c]
                })                                                
                
                if i % 10 == 0 and i > 0:
                    stop = time.time()
                    secs = stop - start
                    print("Execute 10 minutes {} [seconds]".format(int(secs)))
                    start = time.time()         
            with open(f'{self.sequence_path}/{file_id}.avro', 'wb') as out:
                writer(out, self.schema, records)
            
        
if __name__ == '__main__':    
    if sys.argv[1] == 'worker':
        print("Decoding Worker")    
        worker = DecodingWorker('../web_service/ml_models/', '../web_service/images/', '../web_service/sequences/', Redis())
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
        