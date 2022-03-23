import pickle as pkl
import sys
import time
import heapq
import numpy as np 

import tensorflow as tf

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

VERSION    = 'Mar2022v2' 
SEQ_PATH   = f'../web_service/{VERSION}/sequences/'
IMG_PATH   = f'../web_service/{VERSION}/images/'
MODEL_PATH = '../web_service/ml_models/'

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


NEURAL_NOISE_DAMPENING = 0.25
NEURAL_SMOOTH_WIN      = 64


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

    
def ngrams(sequence, n=4, sep=''):
    results = []
    for i in range(n, len(sequence)):
        x = [s.cls for s in sequence[i-n:i]]
        x = sep.join(x)
        results.append(x)
    return results


def match(sequence, db):
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
    result = list(reversed([heapq.heappop(pq) for i in range(0, len(pq))]))
    return [(-1 * d, i) for d, i in result]
    

def query(sequence, sequences, db, k=10):
    ids = match(sequence, db)
    print(ids)
    nn  = knn(sequence, sequences, ids, k)
    print(nn)
    return nn

    
def discovery(sequences, db, k=10):
    neighbors = {}
    densities = {}
    for i, sequence in enumerate(sequences):
        print(f" ... discovery: {i}")
        ids = match(sequence, db)
        nn  = knn(sequence, sequences, ids, k)
        neighbors[i] = nn
        if len(nn) == k:
            densities[i] = 1. / (1 + nn[-1][0])
    return densities, neighbors


def subsequences(sequence, max_len=8):
    n = len(sequence)
    for length in range(1, max_len):
        for i in range(length, n):
            substring = " ".join([s['cls'] for s in sequence[i-length:i]])
            yield substring


class DiscoveryService:
    
    def __init__(self, sequence_path, img_path, limit = None):
        self.sequences = []
        self.keys      = []
        self.samples   = []
        self.decodings = []
        self.densities = {}       
        self.neighbors = {}
        self.substrings = {}
        self.db = {}
        self.decoder = None
        self.lab     = None
        self.reverse = None
        self.label_mapping = None

        self.parse(sequence_path, limit)
        self.setup_discovery()
        self.setup_substrings()        
        self.sequence_path = sequence_path
        self.img_path = img_path
        
    def init_model(self, model_path):
        self.decoder       = load_model(f'{model_path}/decoder_nn.h5', custom_objects = {'Functional' : tf.keras.models.Model})
        self.lab           = pkl.load(open(f"{model_path}/labels.pkl", "rb"))
        self.reverse       = {v:k for k, v in self.lab.items()}
        self.label_mapping = pkl.load(open(f'{model_path}/label_mapping.pkl', 'rb'))
        
    def parse(self, sequence_path, limit):        
        for file in os.listdir(sequence_path):
            print(f" ... reading: {file}")
            if limit is not None and len(self.sequences) >= limit:
                break
            if file.endswith('avro') and not file.startswith('query'):
                with open(f'{sequence_path}/{file}', 'rb') as fo:
                    avro_reader = reader(fo)
                    for record in avro_reader:
                        self.sequences.append(record)

    def setup_substrings(self):
        for i, sequence in enumerate(self.sequences):
            print(f" ... substrings for: {i}")
            for sub in subsequences(sequence['sequence']):
                if sub not in self.substrings:
                    self.substrings[sub] = []
                self.substrings[sub].append(i)
                        
    def setup_discovery(self):
        for key, sequence in enumerate(self.sequences):
            decoded = [DecodedSymbol.from_dict(x) for x in sequence['sequence']]            
            self.decodings.append(decoded)
            for ngram in ngrams(decoded):
                if ngram not in self.db:
                    self.db[ngram] = []
                self.db[ngram].append(key)            
        d, n = discovery(self.decodings, self.db) 
        self.densities  = d        
        self.neighbors  = n                

        self.keys       = list(self.densities.keys())
        self.samples    = np.zeros(len(self.keys))
        scaler          = np.sum(list(self.densities.values()))
        self.samples[0] = self.densities[self.keys[0]] / scaler
        for i in range(1, len(self.keys)):
            self.samples[i] = self.densities[self.keys[i]] / scaler + self.samples[i - 1]         
        
    def sample(self):
        start = 0
        stop  = len(self.samples) - 1
        x     = np.random.uniform()
        while start < stop:
            center = (start + stop) // 2        
            if x > self.samples[center]:
                start = center +  1
            else: 
                stop = center
        region = start
        keys = [neighbor for _, neighbor in self.neighbors[region]]
        nn   = [self.sequences[neighbor] for neighbor in keys]
        return self.sequences[region], nn, keys

    def query_by_file(self, filename):
        name = str(filename).split('/')[-1].split('.')[0]             
        query_id = f"query_{name}"
        audio = raw(filename)
        s = spec(audio)
        start_bound, stop_bound = 0, len(audio)
        dec  = decode(s, self.decoder, self.label_mapping)
        c    = compress_neural(dec, len(s), self.reverse, self.label_mapping)
        img_p = f"{self.img_path}/{query_id}.png"
        plot_neural(s, c, img_p)                
        records = [{                
            "path":     name,
            "start":    start_bound,
            "stop":     stop_bound,
            "sequence": [token.to_dict() for token in c]
        }]                                               
        with open(f'{self.sequence_path}/{query_id}.avro', 'wb') as out:
            writer(out, SCHEMA, records)

        decoded = [DecodedSymbol.from_dict(x) for x in records[0]['sequence']]
        neighbors = query(decoded, self.decodings, self.db)
        keys = [neighbor for _, neighbor in neighbors]
        nn   = [self.sequences[neighbor] for neighbor in keys]        
        return f"{query_id}.png", [s.cls for s in decoded], nn, keys
        
            
    def get(self, region):
        keys = [neighbor for _, neighbor in self.neighbors[region]]
        nn   = [self.sequences[neighbor] for neighbor in keys]
        return self.sequences[region], nn, keys
    
    def find(self, string):
        if string in self.substrings:
            keys = self.substrings[string]            
            nn   = [self.sequences[key] for key in keys]            
            return nn, keys
        else:
            return [], []
    
    
class DecodingWorker:
    
    KEY = 'WDP-DS'
    
    def __init__(self, model_path, image_path, sequence_path, redis):
        self.decoder       = load_model(f'{model_path}/decoder_nn.h5')
        self.lab           = pkl.load(open(f"{model_path}/labels.pkl", "rb"))
        self.reverse       = {v:k for k, v in self.lab.items()}
        self.label_mapping = pkl.load(open(f'{model_path}/label_mapping.pkl', 'rb'))
        self.image_path    = image_path
        self.sequence_path = sequence_path
        
        self.redis         = redis
        self.schema        = parse_schema(SCHEMA)                    
        
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
                if len([c for region in c if region.id > 0]) > 4:
                    plot_neural(s, c, f"{self.image_path}/{file_id}_{start_bound}_{stop_bound}.png")                
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
        worker = DecodingWorker(MODEL_PATH, IMG_PATH, SEQ_PATH, Redis())
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
