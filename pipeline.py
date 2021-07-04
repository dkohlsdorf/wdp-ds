import json
import numpy as np
import pickle as pkl
import sys
import os
import matplotlib.pyplot as plt
import random


from lib_dolphin.audio import *
from lib_dolphin.features import *
from lib_dolphin.eval import *
from lib_dolphin.dtw import *
from lib_dolphin.htk_helpers import *
from collections import namedtuple, Counter

from scipy.io.wavfile import read, write
from tensorflow.keras.models import load_model
from sklearn.cluster import AgglomerativeClustering
from subprocess import check_output


FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 180
FFT_LO       = 60

D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)
RAW_AUDIO    = 5120
T            = int((RAW_AUDIO - FFT_WIN) / FFT_STEP)

CONV_PARAM   = (8, 8, 128)
WINDOW_PARAM = (T, D, 1)
LATENT       = 128
BATCH        = 25
EPOCHS       = 25


def train(label_file, wav_file, noise_file, out_folder="output", perc_test=0.25):
    instances, labels, label_dict = dataset_supervised_windows(
        label_file, wav_file, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, raw_size=RAW_AUDIO)    
 
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
            y_test.append(labels[i])
        else:            
            x_train.append(instances_inp[i])
            y_train.append(labels[i])

    x       = np.stack(instances_inp)[0:len(instances)].reshape(len(instances), T, D, 1)
    x_train = np.stack(x_train).reshape(len(x_train), T, D, 1)
    x_test  = np.stack(x_test).reshape(len(x_test), T, D, 1)
    
    y_train = np.array(y_train)
    y_test  = np.array(y_test)
    model, enc  = classifier(WINDOW_PARAM, LATENT, 5, CONV_PARAM) 
    ae          = auto_encoder(WINDOW_PARAM, enc, LATENT, CONV_PARAM)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=BATCH, epochs=EPOCHS, shuffle=True)
    ae.fit(x=x_train, y=x_train, batch_size=10, epochs=EPOCHS, shuffle=True)
    hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=BATCH, epochs=EPOCHS, shuffle=True)

    enc_filters(enc, CONV_PARAM[-1], "{}/filters.png".format(out_folder))
    plot_tensorflow_hist(hist, "{}/history_train.png".format(out_folder))
    
    x   = enc.predict(x)    

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
    
    model.save('{}/supervised.h5'.format(out_folder))
    enc.save('{}/encoder.h5'.format(out_folder))
    pkl.dump(label_dict, open('{}/labels.pkl'.format(out_folder), "wb"))

    
def clustering(regions, wav_file, folder):
    instances_file   = "{}/instances.pkl".format(folder)
    ids_file         = "{}/ids.pkl".format(folder)
    predictions_file = "{}/predictions.pkl".format(folder)
    distances_file   = "{}/distances.pkl".format(folder)
    clusters_file    = "{}/clusters.pkl".format(folder)
    
    if not os.path.exists(instances_file):
        cls = load_model('{}/supervised.h5'.format(folder))
        enc = load_model('{}/encoder.h5'.format(folder))
        ids, instances, predictions = dataset_unsupervised_regions(
            regions, wav_file, enc, cls, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, T=T)   
        print("#Instances: {}".format(len(instances)))
        pkl.dump(ids, open(ids_file, "wb"))
        pkl.dump(instances, open(instances_file, "wb"))
        pkl.dump(predictions, open(predictions_file, "wb"))
    else:
        instances   = pkl.load(open(instances_file, "rb"))
    if not os.path.exists(distances_file):        
        distances = dtw_distances(instances)
        pkl.dump(distances, open(distances_file, "wb"))    
    else:
        distances = pkl.load(open(distances_file, "rb")) 

    n = 98
    m = len(distances)
    clusters = np.zeros((n, m), dtype=np.int16)
    for perc in range(1, 99):
        i = perc - 1
        if i % 10 == 0:
            print(" ... clustering {}%".format(perc))
        th = np.percentile(distances.flatten(), perc)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=th, affinity="precomputed", linkage="complete")
        clusters[i, :] = clustering.fit_predict(distances)
    pkl.dump(clusters, open(clusters_file, "wb"))


def export(csvfile, wavfile, folder, k, out, min_c = 4):
    print(" ... loading data")
    
    label_file       = "{}/labels.pkl".format(folder)
    ids_file         = "{}/ids.pkl".format(folder)
    predictions_file = "{}/predictions.pkl".format(folder)
    clusters_file    = "{}/clusters.pkl".format(folder)

    ids         = pkl.load(open(ids_file, "rb"))
    clusters    = pkl.load(open(clusters_file, "rb"))[k, :]
    predictions = pkl.load(open(predictions_file, "rb")) 
    label_dict  = pkl.load(open(label_file, "rb"))
    reverse     = dict([(v,k) for k, v in label_dict.items()])
    
    df       = pd.read_csv(csvfile)
    x        = raw(wavfile)
    print(" ... grouping clusters {}".format(np.max(clusters)))
    ranges = []
    for _, row in df.iterrows():
        start = row['starts']
        stop  = row['stops']
        ranges.append((start, stop))
    
    by_cluster  = {}
    ids_cluster = {}
    for i, j in enumerate(ids):
        cluster = clusters[i]
        if cluster not in ids_cluster:
            ids_cluster[cluster] = []
            by_cluster[cluster]  = []
        ids_cluster[cluster].append(i)
        by_cluster[cluster].append(ranges[j])
            
    print(" ... export {} / {}".format(i, len(clusters)))
    unmerged = []
    counts   = [] 
    for c, rng in by_cluster.items():
        label = label_cluster(predictions, ids_cluster[c], reverse)
        if label != "ECHO":
            if len(rng) >= min_c:
                print(" ... export cluster {} {} {} {}".format(c, htk_name(c), len(rng), label))
                counts.append(len(rng))
                audio = []
                for start, stop in rng:
                    for f in x[start:stop]:
                        audio.append(f)
                    for i in range(0, 1000):
                        audio.append(0)
                audio = np.array(audio)
                filename = "{}/{}_{}.wav".format(out, label, c)
                write(filename, 44100, audio.astype(np.int16)) 
            else:
                start, stop = rng[0]
                for f in x[start:stop]:
                    unmerged.append(f)
                for i in range(0, 1000):
                    unmerged.append(0)
    print("Done Export")
    unmerged = np.array(unmerged)
    filename = "{}/unmerged.wav".format(out)
    write(filename, 44100, unmerged.astype(np.int16)) 
    counts.sort(key=lambda x: -x)
    plt.plot(np.log(np.arange(0, len(counts)) + 1), np.log(counts))
    plt.grid(True)
    plt.savefig('{}/{}_log-log.png'.format(out, k))
    plt.close()
    

def htk_converter(file, folder, out):
    print("... convert {} using {} to {}".format(file, folder, out))
    enc      = load_model('{}/encoder.h5'.format(folder))
    audio    = raw(file) 
    spec     = spectrogram(audio, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
    windowed = windowing(spec, T)
    x        = enc.predict(windowed)
    write_htk(x, out)


def dtw_baseline(folder, k = 10, min_c = 4, nn=3, debug = False):
    clusters_file    = "{}/clusters.pkl".format(folder)
    distances_file   = "{}/distances.pkl".format(folder)    
    label_file       = "{}/labels.pkl".format(folder)
    predictions_file = "{}/predictions.pkl".format(folder)
    instances_file   = "{}/instances.pkl".format(folder)
    
    distances   = pkl.load(open(distances_file, "rb"))
    clusters    = pkl.load(open(clusters_file, "rb"))[k, :]
    predictions = pkl.load(open(predictions_file, "rb")) 
    label_dict  = pkl.load(open(label_file, "rb"))
    reverse     = dict([(v,k) for k, v in label_dict.items()])
    instances   = pkl.load(open(instances_file, "rb"))
    
    ids_cluster = {}
    for i, cluster in enumerate(clusters):
        if len(instances[i]) > 0: 
            if cluster not in ids_cluster:
                ids_cluster[cluster] = []
            ids_cluster[cluster].append(i)
            
    train = []
    test  = []
    for c, ids in ids_cluster.items():
        label = label_cluster(predictions, ids, reverse)         
        if label != "ECHO" and len(ids) >= min_c:
            random.shuffle(ids)                
            n_train = int(0.9 * len(ids))
            for i in ids[0:n_train]:
                train.append([i, c])
            for i in ids[n_train:len(ids)]:
                test.append([i, c])
    corr = 0.0
    confusion = []
    ldict = {}
    cur = 0
    for j, true in test:
        candidates = []
        for i, pred in train:            
            candidates.append([pred, distances[i, j]])
        candidates.sort(key = lambda x: x[-1])
        neighbors = [p for p, _ in candidates[0:nn]]
        labels    = [(p, c) for p, c in Counter(neighbors).items()]
        labels.sort(key = lambda x: -x[1])

        if true == labels[0][0]:
            corr += 1
        elif debug:            
            print(true, neighbors, labels)
        if true not in ldict: 
            ldict[true] = cur
            cur += 1
        if labels[0][0] not in ldict:
            ldict[labels[0][0]] = cur
            cur += 1
        confusion.append([ldict[true], ldict[labels[0][0]]])
    conf = np.zeros((cur, cur))
    for i, j in confusion:
        conf[i, j] += 1
    names = [(k, v) for k, v in ldict.items()]
    names.sort(key = lambda x:  x[1])
    names = [k for k, _ in names]
    plot_result_matrix(conf, names, names, "Confusion Window")
    plt.savefig("{}/baseline.png".format(folder))
    plt.close()
    print("Acc: {}".format(corr / len(test)))

    
def htk_train(folder, inputs, states, niter, flat=False):
    print("Prepare project: {}".format(folder))
    out = check_output(["rm", "-rf", folder])
    out = check_output(["mkdir", folder])
    out = check_output(["mkdir", "{}/data".format(folder)])
    htk_export(inputs, "{}/data".format(folder), "{}/clusters.mlf".format(folder), folder)
    files = glob.glob("{}/data/train/*.htk".format(folder))
        
    grammar = simple_grammar("{}/clusters_TRAIN.mlf".format(folder))
    with open("{}/gram".format(folder), 'w') as fp:
        fp.write(grammar + "\n")

    wlist = wordlist("{}/clusters_TRAIN.mlf".format(folder))
    with open("{}/dict".format(folder), 'w') as fp:
        fp.write(wlist + "\n")
        
    print("... flat start")    
    out = check_output(["rm", "-rf", "{}/hmm0".format(folder)])
    out = check_output(["mkdir", "{}/hmm0".format(folder)])
    
    
    if flat:
        hmm = left_right_hmm(states, LATENT, name="proto")
        with open("{}/proto".format(folder), "w") as fp:
            fp.write(hmm)
        out = check_output("HCompV -v {} -T 10 -M {}/hmm0 -m {}/proto".format(FLOOR, folder, folder).split(" ") + files)
        mmf("{}/clusters_TRAIN.mlf".format(folder), "{}/hmm0/proto".format(folder),LATENT, "{}/hmm0/hmm_mmf".format(folder), "{}/list".format(folder))        
    else:
        htk_init("{}/clusters_TRAIN.mlf".format(folder), None, LATENT, "{}/data/train/*.htk".format(folder), folder, LATENT, states, "{}/hmm0".format(folder), "{}/list".format(folder))
    out = check_output("HParse {}/gram {}/wdnet".format(folder, folder).split(" "))

    likelihoods = []
    for i in range(1, niter + 1):
        ll = take_step(folder, i)
        likelihoods.append(ll)
        print("... reest: {} {}".format(i, ll))
    result = htk_eval(folder, niter)
    print(result)
    plt.plot(likelihoods)
    plt.title("Likeihood HMM Mix")
    plt.xlabel("epoch")
    plt.ylabel("ll")
    plt.savefig('{}/ll'.format(folder))
    plt.close()
    htk_confusion("{}/predictions.mlf".format(folder), "{}/confusion_window.png".format(folder))

    
if __name__ == '__main__':
    print("=====================================")
    print("Simplified WDP DS Pipeline")
    print("by Daniel Kyu Hwa Kohlsdorf")
    if len(sys.argv) >= 6 and sys.argv[1] == 'train':            
        labels = sys.argv[2]
        wav    = sys.argv[3]
        noise  = sys.argv[4]
        out    = sys.argv[5]
        train(labels, wav, noise, out)
    elif len(sys.argv) >= 5 and sys.argv[1] == 'clustering':
        labels = sys.argv[2]
        wav    = sys.argv[3]
        out    = sys.argv[4]
        clustering(labels, wav, out)
    elif len(sys.argv) >= 7 and sys.argv[1] == 'export':
        labels   = sys.argv[2]
        wav      = sys.argv[3]
        clusters = sys.argv[4]
        k        = int(sys.argv[5])
        out      =  sys.argv[6]
        export(labels, wav, clusters, k, out)
    elif len(sys.argv) >= 6 and sys.argv[1] == 'htk':
        mode   = sys.argv[2]
        if mode == 'train':
            inputs = sys.argv[3]
            folder = sys.argv[4]
            states = int(sys.argv[5])
            niter  = int(sys.argv[6]) 
            htk_train(folder, inputs, states, niter)
        else:
            audio  = sys.argv[3]
            folder = sys.argv[4]
            htk    = sys.argv[5]
            htk_file = "{}/{}".format(htk, audio.split('/')[-1].replace('.wav', '.htk'))
            htk_converter(audio, folder, htk_file)
    elif len(sys.argv) >= 3 and sys.argv[1] == 'baseline':
          folder = sys.argv[2]
          dtw_baseline(folder)          
    else:
        print("""
            Usage:
                + train:      python pipeline.py train LABEL_FILE AUDIO_FILE NOISE_FILE OUT_FOLDER
                + clustering: python pipeline.py clustering LABEL_FILE AUDIO_FILE OUT_FOLDER
                + export:     python pipeline.py export LABEL_FILE AUDIO_FILE FOLDER K OUT_FOLDER
                + htk:        python pipeline.py htk train FOLDER OUT_HTK STATES ITER
                              python pipeline.py htk convert AUDIO FOLDER OUT_FOLDER 
                + baseline:   python pipeline.py baseline FOLDER
        """)
    print("\n=====================================")
