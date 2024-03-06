import pickle as pkl
import sys
import time
import heapq
import numpy as np 
import json
import tensorflow as tf

import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import polling

from collections import namedtuple
from tensorflow.keras.models import load_model
from scipy.signal import triang

from dbs import *

from lib_dolphin.audio import *
from lib_dolphin.sequential import *
from lib_dolphin.eval import *
from lib_dolphin.discrete import *
from lib_dolphin.parameters import *
from lib_dolphin.extern_index import *

from redis import Redis
from datetime import datetime 

from fastavro import writer, reader, parse_schema
from scipy.io.wavfile import write

ADDR        = 'localhost:50051' 
VERSION     = 'extern_clean' 
SEQ_PATH    = f'../web_service/{VERSION}/sequences/'
IMG_PATH    = f'../web_service/{VERSION}/images/'

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
         },
        {"name" : "proba_ids", "type": {"type" : "array", "items" : "int"}}
    ]
}


def spec(x):
    return spectrogram(x, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
    
def decode(x, decoder, label_mapping, reverse, smoothing=True, win='triang', splitter=2000):
    t, d = x.shape
    result = []
    for i in range(0, t, splitter):
        a = x[i:i+splitter, :]
        a = a.reshape((1,len(a),d,1))
        p = decoder.predict(a, verbose=False).reshape((a.shape[1], label_mapping.n + 1)) 
        result.append(p)
    p = np.concatenate(result)

    if len(p) > NEURAL_SMOOTH_WIN and smoothing:
        for i in range(0, len(p[0])):
            window = triang(NEURAL_SMOOTH_WIN) / np.sum(triang(NEURAL_SMOOTH_WIN)) 
            p[:, i] = np.convolve(p[:, i], window, mode='same')
    p[:, 0] *= NEURAL_NOISE_DAMPENING
    for i in range(1, len(p[0])):
        dc = i2name(i, reverse, label_mapping)
        if dc in NEURAL_LABEL_DAMPENING:
            df = NEURAL_LABEL_DAMPENING[dc]
            print(f" ... dampen {dc} by {df}")
            p[:, i] *= df
    
    local_c = p.argmax(axis=1)
    local_p = p.max(axis=1)                    
    local_c = [reject(local_c[i], local_p[i], NEURAL_REJECT[i2name(local_c[i], reverse, label_mapping)])
               for i in range(len(local_c))]

    return local_c, p

    
def ngrams(sequence, n=2, sep=''):
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
    
    
def overlap(x1, x2, y1, y2, e1, e2):
    return e1 == e2 and max(x1.start, y1.start) <= min(x2.stop, y2.stop)
    
    
def knn(sequence, sequences, ids, eids = None, k=10):
    pq = []
    for i in range(0, min(k, len(ids))):        
        if eids is not None and not overlap(sequence[0], sequence[-1], sequences[ids[i]][0], sequences[ids[i]][-1], eids[0], eids[1][ids[i]]):
            d = levenstein(labels(sequence), labels(sequences[ids[i]]))
            heapq.heappush(pq, (-d, ids[i]))
        elif eids is not None:
            print(f"Overlap Rejected {eids[0]} {eids[1][ids[i]]} [{sequence[0].start}, {sequence[-1].stop}] :: [{sequences[ids[i]][0].start}, {sequences[ids[i]][-1].stop}]")
        else:
            d = levenstein(labels(sequence), labels(sequences[ids[i]]))
            heapq.heappush(pq, (-d, ids[i]))
    if len(pq) == 0:
        return []
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
    nn  = knn(sequence, sequences, ids, None, k)
    print(nn)
    return nn

    
def discovery(sequences, db, eids, k=10):
    neighbors = {}
    densities = {}
    for i, sequence in enumerate(sequences):
        print(f" ... discovery: {i}")
        ids = match(sequence, db)
        nn  = knn(sequence, sequences, ids, (eids[i], eids), k)
        if len(nn) == k:
            neighbors[i] = nn
            densities[i] = 1. / (1 + nn[-1][0])
        else:
            neighbors[i] = nn
            densities[i] = 1e-8
            
    return densities, neighbors


def subsequences(sequence, max_len=8):
    n = len(sequence)
    for length in range(1, max_len):
        for i in range(length, n):
            substring = " ".join([s['cls'] for s in sequence[i-length:i]])
            yield substring
            

class DiscoveryService:
    
    def __init__(self, sequence_path, img_path, limit = None):
        self.sequences     = []
        self.keys          = []
        self.samples       = []
        self.decodings     = []
        self.encounter_ids = []

        # TODO ts id extern index -> sequence 
        self.inverted_idx = {}
        
        self.densities  = {}       
        self.neighbors  = {}
        self.substrings = {}
        self.db         = {}
        
        self.decoder       = None
        self.lab           = None
        self.reverse       = None
        self.label_mapping = None

        self.parse(sequence_path, limit)
        self.setup_discovery()
        self.setup_substrings()        
        self.setup_inverted()
        self.sequence_path = sequence_path
        self.img_path = img_path    
        
    def init_model(self, model_path):
        self.decoder       = load_model(f'{model_path}/decoder_nn.h5', custom_objects = {'Functional' : tf.keras.models.Model})
        self.lab           = pkl.load(open(f"{model_path}/labels.pkl", "rb"))
        self.reverse       = {v:k for k, v in self.lab.items()}
        self.label_mapping = pkl.load(open(f'{model_path}/label_mapping.pkl', 'rb'))
        load(ADDR, VERSION)
        
    def parse(self, sequence_path, limit):            
        for file in os.listdir(sequence_path):
            eid = file.replace('.avro', '')
            print(f" ... reading: {file} {eid}")
            if limit is not None and len(self.sequences) >= limit:
                break
            if file.endswith('avro') and not file.startswith('query'):
                with open(f'{sequence_path}/{file}', 'rb') as fo:
                    avro_reader = reader(fo)
                    for record in avro_reader:
                        self.sequences.append(record)
                        self.encounter_ids.append(eid)                        
                        
    def setup_substrings(self):
        for i, sequence in enumerate(self.sequences):
            print(f" ... substrings for: {i}")
            for sub in subsequences(sequence['sequence']):
                if sub not in self.substrings:
                    self.substrings[sub] = []
                self.substrings[sub].append(i)

    def setup_inverted(self):
        for i, sequence in enumerate(self.sequences):
            for ts_id in sequence['proba_ids']:
                self.inverted_idx[ts_id] = i 
        
    def setup_discovery(self):
        for key, sequence in enumerate(self.sequences):
            decoded = [DecodedSymbol.from_dict(x) for x in sequence['sequence']]            
            self.decodings.append(decoded)
            for ngram in ngrams(decoded):
                if ngram not in self.db:
                    self.db[ngram] = []
                self.db[ngram].append(key)            
        d, n = discovery(self.decodings, self.db, self.encounter_ids) 
        print(f"Done discovery {len(d)} {len(n)}")
        
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
    
    def query_by_file(self, filename, relax=False):
        name = str(filename).split('/')[-1].split('.')[0]             
        query_id = f"query_{name}"
        audio = raw(filename)
        s = spec(audio)
        plottable = spectrogram(audio, 0, FFT_WIN // 2, FFT_WIN, FFT_STEP)
        start_bound, stop_bound = 0, len(audio)
        dec, probs = decode(s, self.decoder, self.label_mapping, self.reverse)
        c    = compress_neural(dec, len(s), self.reverse, self.label_mapping)
        img_p = f"{self.img_path}/{query_id}.png"
        plot_neural(plottable, c, img_p)

        n = len(probs)
        probas = []
        for i in range(100, n, 10):
            probas.append(probs[i-100:i])    
        
        records = [{                
            "path":      name,
            "start":     start_bound,
            "stop":      stop_bound,
            "sequence":  [token.to_dict() for token in c],
            "proba_ids": []
        }]                                               
        with open(f'{self.sequence_path}/{query_id}.avro', 'wb') as out:
            writer(out, SCHEMA, records)

        decoded = [DecodedSymbol.from_dict(x) for x in records[0]['sequence']]
        if relax:
            neighbors = find_relaxed(ADDR, VERSION, probas, self.inverted_idx)
        else:
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
    JSON_KEY = "WDP-DS_JSON"
    
    def __init__(self, model_path, image_path, sequence_path, redis=None):
        self.decoder       = load_model(f'{model_path}/decoder_nn.h5')
        self.lab           = pkl.load(open(f"{model_path}/labels.pkl", "rb"))
        self.reverse       = {v:k for k, v in self.lab.items()}
        self.label_mapping = pkl.load(open(f'{model_path}/label_mapping.pkl', 'rb'))
        self.image_path    = image_path
        self.sequence_path = sequence_path
        
        self.redis         = redis
        self.schema        = parse_schema(SCHEMA)                    

    def process(self, filename):
        assert filename.endswith('wav')    
        x         = raw(filename)
        plottable = spectrogram(x, 0, FFT_WIN // 2, FFT_WIN, FFT_STEP)
        s         = spec(x)
        _, probs = decode(s, self.decoder, self.label_mapping, self.reverse)
        return x, plottable, probs

    def to_json(self, fname):
        img_path = fname.replace(".wav", ".png")
        json_path = fname.replace(".wav", ".json")
        fileid = fname.split("/")[-1].replace(".wav", "")
        print(f"Convert {fname} to json file {json_path} and img {img_path}")

        _, spec, probs = self.process(fname)
        print(f" ... sizes {spec.shape} {probs.shape}")
        matplotlib.image.imsave(img_path, BIAS - spec.T * SCALER, cmap='gray')
        
        output = {
            "spec"  : fileid,
            "probs" : probs.tolist()
        }
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        db = AlignmentDB()        
        db.finish_file(fileid)

        
    def json_work(self):        
        result = self.redis.lpop(DecodingWorker.JSON_KEY)
        if result is None:
            print("\t ... no work")
        else:
            fname = result.decode('utf-8')
            print(f"... {result} {type(result)}")
            self.to_json(fname)
            
        
    def work(self):
        now = datetime.now()        
        result = self.redis.lpop(DecodingWorker.KEY)
        
        print(f'.. Check for work {now} {result}')        
        if result is not None:
            if result == b'reindex':
                reindex(ADDR, VERSION)
                return
            
            records = []
            
            filename = result
            file_id = str(filename).split('/')[-1].split('.')[0]             
            print(f'.. Work: {filename} {file_id}')
            regions, bounds, audio_file = split(filename)
            start = time.time()

            for i in range(len(regions)):
                s         = spec(regions[i])
                plottable = spectrogram(regions[i], 0, FFT_WIN // 2, FFT_WIN, FFT_STEP)
                start_bound, stop_bound = bounds[i] 
                dec, probs = decode(s, self.decoder, self.label_mapping, self.reverse)
                c = compress_neural(dec, len(s), self.reverse, self.label_mapping)
                print(f" ... {i}: {len(dec)} {len(c)} {len([c for region in c if region.id > 0])}")
                if len([c for region in c if region.id > 0]) > 8:
                    png_file   = f"{self.image_path}/{file_id}_{start_bound}_{stop_bound}.png"
                    audio_file = f"{self.image_path}/{file_id}_{start_bound}_{stop_bound}.wav"
                    raven_tab  = f"{self.image_path}/{file_id}_{start_bound}_{stop_bound}.txt"
                    write(audio_file, 44100, regions[i])
                    raven(raven_tab, c)

                    n = len(probs)
                    probas = []                
                    for i in range(100, n, 50):
                        probas.append(probs[i-100:i])
                    ids = insert_all(probas, ADDR)
                    
                    plot_neural(plottable, c, png_file)
                    records.append({                
                        "path":     str(filename),
                        "start":    start_bound,
                        "stop":     stop_bound,
                        "sequence": [token.to_dict() for token in c],
                        "proba_ids": ids
                    })                                                
                
                if i % 10 == 0 and i > 0:
                    stop = time.time()
                    secs = stop - start
                    print("Execute 10 minutes {} [seconds]".format(int(secs)))
                    start = time.time()         
            with open(f'{self.sequence_path}/{file_id}.avro', 'wb') as out:
                writer(out, self.schema, records)

                
def transitions(sequence_path, output):
    sequences = []
    for file in os.listdir(sequence_path):
        eid = file.replace('.avro', '')
        print(f" ... reading: {file} {eid}")
        if file.endswith('avro') and not file.startswith('query'):
            with open(f'{sequence_path}/{file}', 'rb') as fo:
                avro_reader = reader(fo)
                for record in avro_reader:
                    sequences.append(record)
    unigrams = []
    for sequence in sequences:
        for symbol in sequence['sequence']:
            if symbol['cls'].startswith('_'):
                unigrams.append('_')
            else:
                unigrams.append(symbol['cls'])
    
    unigrams = sorted(list(set(unigrams)))
    idx = {unigram : i for i, unigram in enumerate(unigrams)}
    n = max(idx.values()) + 1
    bigrams = np.zeros((n, n))
    for sequence in sequences:
        for i in range(1, len(sequence['sequence'])):
            n0 = sequence['sequence'][i - 1]['cls'] 
            n1 = sequence['sequence'][i]['cls']
            if not n0.startswith('_') and not n1.startswith('_'):
                i = idx[n0]
                j = idx[n1]
                bigrams[i][j] += 1
    plot_result_matrix(bigrams, unigrams, unigrams, "transitions")
    plt.savefig(output)
    plt.close()
        

if __name__ == '__main__':    
    if sys.argv[1] == 'worker':
        print("Decoding Worker")   
        worker = DecodingWorker(MODEL_PATH, IMG_PATH, SEQ_PATH, Redis())
        polling.poll(lambda: worker.work(), step=5, poll_forever=True)        
    elif sys.argv[1] == 'enqueue':
        print('Batch Enqueue')
        folder = sys.argv[2]
        r = Redis()
        r.lpush(DecodingWorker.KEY, 'reindex')
        for filename in os.listdir(folder):
            if not filename.startswith('.') and (filename.endswith('.wav') or filename.endswith('.WAV')):
                path = f'{folder}/{filename}'
                print(f" .. Enqueue: {path}")
                r.lpush(DecodingWorker.KEY, path)
    elif sys.argv[1] == 'transitions':
        print("Compute Transitions")
        output = sys.argv[2]
        transitions(SEQ_PATH, output)
    elif sys.argv[1] == 'convert_json':
        if(len(sys.argv)) == 3:            
            fname = sys.argv[2]
            worker = DecodingWorker(MODEL_PATH, IMG_PATH, SEQ_PATH, None)
            worker.to_json(fname)
    elif sys.argv[1] == 'json_converter':
        worker = DecodingWorker(MODEL_PATH, IMG_PATH, SEQ_PATH, Redis())
        polling.poll(lambda: worker.json_work(), step=5, poll_forever=True)
            
