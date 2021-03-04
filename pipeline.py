import numpy as np
import pickle as pkl
import nmslib 
import sys
import os

from lib_dolphin.audio import *
from lib_dolphin.features import *
from lib_dolphin.interest_points import *

from collections import namedtuple

from sklearn.cluster import * 


LABELS = set([
    'EC_FAST', 
    'BP_MED',
    'WSTL_DOWN',
    'WSTL_UP'
])


FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 200
FFT_LO       = 20

D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)
RAW_AUDIO    = 5120
T            = int((RAW_AUDIO - FFT_WIN) / FFT_STEP)


CONV_PARAM   = (8, 8, 128)
WINDOW_PARAM = (T, D, 1)
LATENT       = 64
BATCH        = 25
EPOCHS       = 50

N_DIST       = 10000
PERC_TH      = 5

IP_RADIUS    = 6
IP_DB_TH     = 3

KNN          = 25
PROC_BATCH   = 1000    


def train(label_file, wav_file, out_folder="output", labels = LABELS, perc_test=0.1):
    _, instances, labels, label_dict = dataset_supervised(label_file, wav_file, labels)
    x_train = []
    x_test = []
    for i in range(0, len(instances)):
        if np.random.uniform() < perc_test:
            x_test.append(instances[i])
        else:
            x_train.append(instances[i])
            
    x       = np.stack(instances).reshape(len(instances), T, D, 1)
    x_train = np.stack(x_train).reshape(len(x_train), T, D, 1)
    x_test  = np.stack(x_test).reshape(len(x_test), T, D, 1)
            
    ae, enc, dec = auto_encoder(WINDOW_PARAM, LATENT, CONV_PARAM)
    ae.compile(optimizer='adam', loss='mse', metrics=['mse'])
    ae.fit(x=x_train, y=x_train, validation_data=(x_test, x_test), batch_size=BATCH, epochs=EPOCHS, shuffle=True)
    
    x   = enc.predict(x)
    
    distances = []
    for i in range(N_DIST):
        idx  = np.random.randint(len(x))
        idy  = np.random.randint(len(x))
        if idx != idy:
            dist = np.sqrt(np.sum(np.square(x[idx] - x[idy])))
            distances.append(dist)
    th = np.percentile(distances, PERC_TH)
    print("Threshold: {}".format(th))    
    
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=th, affinity='euclidean', linkage='complete')
    c   = agg.fit_predict(x)
    
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(x)
    index.createIndex({'post': 2}, print_progress=True)
    
    enc.save('{}/encoder.h5'.format(out_folder))
    pkl.dump((th, c, labels, label_dict), open("{}/labels.pkl".format(out_folder), "wb"))
    nmslib.saveIndex(index, '{}/index'.format(out_folder))
    

def label(x):
    counts = {}
    for c in x:
        if c not in counts:
            counts[c] = 0
        counts[c] += 1
    max_count = 0
    max_class = 0
    for c, i in counts.items():
        if i > max_count:
            max_count = i
            max_class = c
    p = max_count / len(x)
    return max_class, p    
    
    
Model = namedtuple("Model", "clusters labels label_dict index encoder th")


def process_batch(batch, batch_off, model, reverse):
    batch = np.stack(batch)
    x     = model.encoder.predict(batch)
    for xid in range(0, len(x)):
        ids, d = model.index.knnQuery(x[xid], k = KNN)
    
        clusters    = [model.clusters[xi] for xi in ids]
        labels      = [model.labels[xi] for xi in ids]

        cluster, pc = label(clusters)
        labi, pl    = label(labels) 
        lab         = reverse[labi]
        start = batch_off[xid] 
        stop  = batch_off[xid] + RAW_AUDIO
        if d[-1] < model.th:
            yield [lab, cluster, start, stop]

    
def apply_model(file, model):   
    x = raw(file)
    s = spectrogram(x)
    r = T // 2
    
    offsets = []
    patches = []
    for t, f in interest_points(s, IP_RADIUS, IP_DB_TH):
        start_t = t - r
        offset  = start_t * FFT_STEP
        if len(offsets) == 0 or offsets[-1] < offset:
            if t > r and t < len(x) - r:
                offsets.append(offset)
                spec  = s[t - r : t + r]
                mu    = np.mean(spec)
                sigma = np.std(spec) + 1.0
                spec  = (spec - mu) / sigma
                patches.append(spec)
                
    offsets = [o for o,p in zip(offsets, patches) if p.shape == (T, D)]
    patches = np.stack([p for p in patches if p.shape == (T, D)])
    patches = patches.reshape((len(patches), T, D, 1))

    reverse   = dict([(v, k) for k, v in model.label_dict.items()])
    anno      = []
    batch     = []
    batch_off = []
    for i, offset in enumerate(offsets):
        batch.append(patches[i, :,:,:])
        batch_off.append(offset)        
        if len(batch) == PROC_BATCH:
            for annotation in process_batch(batch, batch_off, model, reverse):
                anno.append(annotation)
            batch = []
            batch_off = []    
    for annotation in process_batch(batch, batch_off, model, reverse):
        anno.append(annotation)            
    return anno

                  
def apply_model_files(files, out_folder="output"):
    index = nmslib.init(method ='hnsw', space='l2')
    nmslib.loadIndex(index, '{}/index'.format(out_folder))
    
    th, c, labels, label_dict = pkl.load(open("{}/labels.pkl".format(out_folder), "rb"))
    enc = load_model('{}/encoder.h5'.format(out_folder))   
    model = Model(c, labels, label_dict, index, enc, 5.0)
    for file in files:
        annotations = apply_model(file, model)
        name = "{}/{}".format(out_folder, file.split("/")[-1].replace('.wav', '.csv'))
        print("Processing {} to {}".format(file, name))
        df = pd.DataFrame({
            'labels':  [label   for label, _, _, _ in annotations],
            'cluster': [cluster for _, cluster, _, _ in annotations],
            'start':   [start   for _, _, start, _ in annotations],
            'stop':    [stop    for _, _, _, stop in annotations]
        })
        df.to_csv(name, index=False)
    
                  
if __name__ == '__main__':
    print("=====================================")
    print("Simplified WDP DS Pipeline")
    print("by Daniel Kyu Hwa Kohlsdorf")
    if len(sys.argv) == 4 and sys.argv[1] == 'train':            
            labels = sys.argv[2]
            wav    = sys.argv[3]
            train(labels, wav)
    elif len(sys.argv) == 3 and sys.argv[1] == 'test':        
        path = sys.argv[2]
        apply_model_files(["{}/{}".format(path, filename) for filename in os.listdir(path) if filename.endswith('.wav')])
    else:
        print("""
            Usage:
                + train: python pipeline.py train LABEL_FILE AUDIO_FILE
                + test:  python pipeline.py test FOLDER
        """)
    print("=====================================")
