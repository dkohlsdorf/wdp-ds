import json
import numpy as np
import pickle as pkl
import nmslib 
import sys
import os

from lib_dolphin.audio import *
from lib_dolphin.features import *
from lib_dolphin.interest_points import *
from lib_dolphin.eval import *
from lib_dolphin.sequencing import *

from collections import namedtuple

from scipy.io.wavfile import read, write
from sklearn.cluster import * 


LABELS = set([
    'EC_FAST', 
    'BP_MED',
    'WSTL_DOWN',
    'WSTL_UP'
])


FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 180
FFT_LO       = 60

D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)
RAW_AUDIO    = 5120
T            = int((RAW_AUDIO - FFT_WIN) / FFT_STEP)

CONV_PARAM   = (8, 8, 128)
WINDOW_PARAM = (T, D, 1)
LATENT       = 64
BATCH        = 25
EPOCHS       = 25

N_DIST       = 10000
PERC_TH      = 25 

IP_RADIUS    = 6
IP_DB_TH     = 1.0

KNN          = 25
PROC_BATCH   = 1000    

SUPERVISED   = True
PLOT_POINTS  = False
MIN_COUNT    = 3
TH_NW_PERC   = 50
GAP          = -1.0


def train(label_file, wav_file, noise_file, out_folder="output", labels = LABELS, perc_test=0.25):
    windows, instances, labels, label_dict = dataset_supervised(
        label_file, wav_file, labels, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, raw_size=RAW_AUDIO)    
    
    noise_label  = np.max([i for _, i in label_dict.items()]) + 1
    label_dict['NOISE'] = noise_label
    label_counts = {}
    for i in labels:
        if i in label_counts:
            label_counts[i] += 1
        else:
            label_counts[i] =1
    max_count = np.max([c for _, c in label_counts.items()])
    print("Count: {}".format(max_count))
    print("Labels: {}".format(label_dict))
    noise = spectrogram(raw(noise_file), lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP)
    instances_inp = []
    for i in range(0, len(instances)):
        stop  = np.random.randint(36, len(noise))
        start = stop - 36
        instances_inp.append((instances[i] + noise[start:stop, :]) / 2.0)

    n_noise = 0
    for i in range(0, max_count):
        stop  = np.random.randint(36, len(noise))
        start = stop - 36        
        instances_inp.append(noise[start:stop, :])
        labels.append(noise_label)
        n_noise += 1
    print("Added: {} ".format(n_noise ))
    visualize_dataset(instances, "{}/dataset.png".format(out_folder))
    visualize_dataset(instances_inp, "{}/dataset_noisy.png".format(out_folder))

    y_train = []
    y_test  = []
    x_train = []
    x_test  = []
    for i in range(0, len(instances_inp)):
        if np.random.uniform() < perc_test:
            x_test.append(instances_inp[i])
            if SUPERVISED:
                y_test.append(labels[i])
            else:
                y_test.append(instances_inp[i])
        else:            
            x_train.append(instances_inp[i])
            if SUPERVISED:
                y_train.append(labels[i])
            else:
                y_train.append(instances_inp[i])

    x       = np.stack(instances_inp)[0:len(instances)].reshape(len(instances), T, D, 1)
    x_train = np.stack(x_train).reshape(len(x_train), T, D, 1)
    x_test  = np.stack(x_test).reshape(len(x_test), T, D, 1)
    
    if SUPERVISED:
        y_train = np.array(y_train)
        y_test  = np.array(y_test)
        model, enc  = classifier(WINDOW_PARAM, LATENT, 5, CONV_PARAM) 
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        y_train = np.stack(y_train).reshape(len(y_train), T, D, 1)
        y_test  = np.stack(y_test).reshape(len(y_test), T, D, 1)

        model, enc, dec = auto_encoder(WINDOW_PARAM, LATENT, CONV_PARAM)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=BATCH, epochs=EPOCHS, shuffle=True)
    
    enc_filters(enc, CONV_PARAM[-1], "{}/filters.png".format(out_folder))
    plot_tensorflow_hist(hist, "{}/history_train.png".format(out_folder))
    
    x   = enc.predict(x)    

    if SUPERVISED:
        n = len(label_dict)
        label_names = ["" for i in range(n)]
        for l, i in label_dict.items():
            label_names[i] = l
        prediction_test = model.predict(x_test)
        confusion = np.zeros((n,n))
        for i in range(len(y_test)):
            pred = np.argmax(prediction_test[i])
            confusion[y_test[i], pred] += 1
        plot_result_matrix(confusion, label_names, label_names, "confusion")
        plt.savefig('{}/confusion_type.png'.format(out_folder))
        plt.close()
    else:
        reconstruct(model, instances, "{}/reconstruction.png".format(out_folder))
    
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
    
    whitelist = export_audio(c, labels[0:len(instances)], windows, label_dict, out_folder)
    print(whitelist)

    print("Shape: {}, Labels: {}, C:{} ".format(x.shape, len(labels), len(c)))
    n      = len(x)
    x      = np.stack([x[i]   for i in range(0, n) if c[i] in whitelist])
    labels = [labels[i]       for i in range(0, n) if c[i] in whitelist]
    c      = [whitelist[c[i]] for i in range(0, n) if c[i] in whitelist]
    print("Shape: {}, Labels: {}, C:{} ".format(x.shape, len(labels), len(c)))

    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(x)
    index.createIndex({'post': 2}, print_progress=True)

    if SUPERVISED:
        model.save('{}/supervised.h5'.format(out_folder))
    else:
        model.save('{}/ae.h5'.format(out_folder))
    enc.save('{}/encoder.h5'.format(out_folder))
    pkl.dump((th, c, labels, label_dict, x), open("{}/labels.pkl".format(out_folder), "wb"))
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
    assignment = []
    for k in counts.keys():
        assignment.append((k, counts[k] / len(x)))
    return max_class, p, assignment
    
    
Model = namedtuple("Model", "clusters labels label_dict index encoder classifier")


def process_batch(batch, batch_off, model, reverse):
    batch = np.stack(batch)
    x     = model.encoder.predict(batch)
    if SUPERVISED:
        y = model.classifier.predict(batch)
    for xid in range(0, len(x)):
        ids, d = model.index.knnQuery(x[xid], k = KNN)
    
        clusters    = [model.clusters[xi] for xi in ids]
        labels      = [model.labels[xi] for xi in ids]
        
        cluster, pc, assignments = label(clusters)
        labi, pl, _              = label(labels) 
        lab                      = reverse[labi]
        start = batch_off[xid] 
        stop  = batch_off[xid] + RAW_AUDIO
        if SUPERVISED:
            argi  = np.argmax(y[xid])
            prob  = y[xid, argi]
            lab_y = reverse[argi]
        else:
            lab_y = None
            prob  = None
        yield [lab_y, lab, cluster, start, stop, prob, assignments]

    
def apply_model(file, model):   
    x  = raw(file)
    s  = spectrogram(x, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP)
    r = T // 2
    
    offsets = []
    patches = []
    ip      = []
    for t, f in interest_points(s, IP_RADIUS, IP_DB_TH):
        start_t = t - r
        offset  = start_t * FFT_STEP
        if len(offsets) == 0 or offsets[-1] < offset:
            if t > r and t < len(x) - r:
                offsets.append(offset)
                spec  = s[t - r : t + r]
                patches.append(spec)
                ip.append({"t": t, "f": f + FFT_LO})
                
    offsets = [o for o,p in zip(offsets, patches) if p.shape == (T, D)]
    patches = [p for p in patches if p.shape == (T, D)]
    if len(patches) == 0:
        return None
    patches = np.stack(patches)
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
    if len(batch) > 0:
        for annotation in process_batch(batch, batch_off, model, reverse):
            anno.append(annotation)            
    return anno, ip

                  
def apply_model_files(files, out_folder="output"):
    index = nmslib.init(method ='hnsw', space='l2')
    nmslib.loadIndex(index, '{}/index'.format(out_folder))
    
    _, c, labels, label_dict, _ = pkl.load(open("{}/labels.pkl".format(out_folder), "rb"))
    enc = load_model('{}/encoder.h5'.format(out_folder))
    if SUPERVISED:
        classifier = load_model('{}/supervised.h5'.format(out_folder))
    else:
        classifier = None
        
    model = Model(c, labels, label_dict, index, enc, classifier)
        
    csv = []
    for file in files:
        res = apply_model(file, model)
        if res is not None:
            annotations, ip = res
            name = "{}/{}".format(out_folder, file.split("/")[-1].replace('.wav', '.csv'))        
            print("Processing {} to {}".format(file, name))
            df = pd.DataFrame({
                'labels':     [label   for label, _, _, _, _, _, _   in annotations],
                'knn':        [label   for _, label, _, _, _, _, _   in annotations],
                'cluster':    [cluster for _, _, cluster, _, _, _, _ in annotations],
                'start':      [start   for _, _, _, start, _, _, _   in annotations],
                'stop':       [stop    for _, _, _, _, stop, _, _    in annotations],
                'prob':       [prob    for _, _, _, _, _, prob, _    in annotations],
                'density':    [dens    for _, _, _, _, _, _, dens    in annotations]
            })
            dec, before_smoothing = decoded(df)
            smooth(dec, before_smoothing, df, name)
            csv.append(name)
            
            
def string(r):
        return " ".join(["{}{}".format(s.type[0], s.id) for s in r])

    
def aligned(input_path, path_out, min_len = 0, use_pam = False):    
    savefile = "{}/aligned_prep.pkl".format(path_out)    
    if os.path.exists(savefile):
        all_regions, distance = pkl.load(open(savefile, 'rb'))
        sequences = [region[1] for region in all_regions]
    else:
        all_regions = []
        for file in os.listdir(input_path):
            if file.endswith('.csv'):
                path  = "{}/{}".format(input_path, file)
                audio = path.replace('.csv', '.wav')
                df    = pd.read_csv(path)    
                for r in regions(df):
                    if len(r) > min_len:
                        all_regions.append((audio, r))
        print("#Regions: {}".format(len(all_regions)))
        sequences = [region[1] for region in all_regions]
        if use_pam:
            _, c, _, _, x = pkl.load(open("{}/labels.pkl".format(path_out), 'rb'))
            inter_class   = pam(c, x)
            print("PAM: 5 {} :: 50 {} :: 95 {} :: max {}".format(np.percentile(inter_class, 5), np.percentile(inter_class, 50), np.percentile(inter_class, 95), np.max(inter_class)))
            plt.imshow(inter_class)
            plt.savefig('{}/pam.png'.format(path_out))
            plt.close()
            distance = distances(sequences, GAP, inter_class, False)
        else:
            distance  = distances(sequences, GAP, None, False)
        pkl.dump((all_regions, distance), open(savefile, 'wb'))
    th = np.percentile(distance, TH_NW_PERC)
    print("Threshold: {}".format(th))
    distance_plots(distance, path_out)

    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=th).fit_predict(distance)

    counts = {}
    for c in clustering:
        if c not in counts:
            counts[c] = 0
        counts[c] += 1   

    n_regions = 0
    clustered = {}
    names  = {}
    cur    = 0
    for i, c in enumerate(clustering):
        if c not in names:
            if counts[c] > MIN_COUNT:
                names[c] = cur
                cur += 1
        if c not in clustered: 
            clustered[c] = []        
        clustered[c].append([all_regions[i][0], all_regions[i][1][0].start, all_regions[i][1][-1].stop, string(all_regions[i][1])])
        n_regions += 1

    print("#Clusters: {}".format(len(names)))
    decoded_plots(clustered, names, counts, path_out, IP_DB_TH, IP_RADIUS, MIN_COUNT, PLOT_POINTS)
    sequence_cluster_export(clustered, names, counts, path_out, MIN_COUNT)
    

def slice_intersting(audio_file, out, processing_window = 10 * 44100):
    x = raw(audio_file)
    n = len(x)
    regions = []
    for i in range(processing_window, n, processing_window // 2):        
        s        = spectrogram(x[i - processing_window:i], lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP)
        n_points = len([x for x in interest_points(s, IP_RADIUS, IP_DB_TH)])
        region   = (i-processing_window, i, n_points)
        regions.append(region)
    if len(regions) > 0:
        th = np.percentile([n for _, _, n in regions], 95)
        print("Activity Threshold: {} of {} regions".format(th, len(regions)))
        connected = []
        last_active = 0
        recording = False
        for i in range(1, len(regions)):
            if regions[i][2] >= th and not recording:
                recording = True
                last_active = i
            elif regions[i][2] < th and recording:
                print("DETECTED: {} : {}".format(last_active, i))
                connected.append([regions[last_active][0], regions[i - 1][1]])
                recording = False
        print("Detected Regions: {}".format(len(connected)))
        for start, stop in connected:
            name = "{}_{}.wav".format(audio_file.split("/")[-1].replace('.wav', ''), start)
            write('{}/{}'.format(out, name), 44100, x[start:stop].astype(np.int16)) 

        
if __name__ == '__main__':
    print("=====================================")
    print("Simplified WDP DS Pipeline")
    print("by Daniel Kyu Hwa Kohlsdorf")
    if len(sys.argv) == 6 and sys.argv[1] == 'train':            
            labels = sys.argv[2]
            wav    = sys.argv[3]
            noise  = sys.argv[4]
            out    = sys.argv[5]            
            train(labels, wav, noise, out)
    elif len(sys.argv) == 4 and sys.argv[1] == 'test':        
        path = sys.argv[2]
        out  = sys.argv[3]
        wavfiles = ["{}/{}".format(path, filename) for filename in os.listdir(path) if filename.endswith('.wav')]    
        apply_model_files(wavfiles, out)
    elif len(sys.argv) == 4 and sys.argv[1] == 'aligned':        
        path = sys.argv[2]
        out  = sys.argv[3]
        aligned(path, out)
    elif len(sys.argv) == 4 and sys.argv[1] == 'slice':
            audio = sys.argv[2]
            out   = sys.argv[3]
            for filename in os.listdir(audio):                
                if filename.endswith('.wav'):
                    print("Slicing: {}".format(filename))
                    path = "{}/{}".format(audio, filename)
                    slice_intersting(path, out)
    else:
        print("""
            Usage:
                + train:     python pipeline.py train LABEL_FILE AUDIO_FILE NOISE_FILE OUT_FOLDER
                + test:      python pipeline.py test FOLDER OUT
                + aligned:   python pipeline.py aligned FOLDER OUT
                + slice:     python pipeline.py slice AUDIO_FILE OUT_FOLDER
        """)
    print("\n=====================================")
