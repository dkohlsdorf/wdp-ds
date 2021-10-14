import json
import numpy as np
import pickle as pkl
import sys
import os
import matplotlib.pyplot as plt
import random

import nmslib

from lib_dolphin.audio import *
from lib_dolphin.features import *
from lib_dolphin.eval import *
from lib_dolphin.dtw import *
from lib_dolphin.htk_helpers import *
from lib_dolphin.sequential import *

from collections import namedtuple, Counter

from scipy.io.wavfile import read, write
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from sklearn.cluster import AgglomerativeClustering, KMeans
from kneed import KneeLocator

from subprocess import check_output

NEURAL_NOISE_DAMPENING=0.25
NEURAL_SMOOTH_WIN=32
NEURAL_SIZE_TH=16

FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 230
FFT_LO       = 100

D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)
RAW_AUDIO    = 5120
T            = int((RAW_AUDIO - FFT_WIN) / FFT_STEP)

CONV_PARAM   = [
    (8, 8,  32),
    (4, 16, 32),
    (2, 32, 32),
    (1, 64, 32),
    (8,  4, 32),
    (16, 4, 32),
    (32, 4, 32)
]
N_BANKS = len(CONV_PARAM)
N_FILTERS = np.sum([i for _, _, i in CONV_PARAM])

WINDOW_PARAM = (T, D, 1)
LATENT       = 128
BATCH        = 25
EPOCHS       = 10


def compute_bic(kmeans, X):
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    m = kmeans.n_clusters
    n = np.bincount(labels)
    N, d = X.shape    
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * np.log(N) * (d+1)
    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    return(BIC)

    
def cluster_model(data, out_folder, label, min_k=2, max_k=26): 
    scores = []
    models = []
    for k in range(min_k, max_k):
        km = KMeans(n_clusters=k)
        km.fit(data)
        bic = compute_bic(km, data)
        scores.append(bic)
        models.append(km)
    kn = KneeLocator(np.arange(len(scores)), scores, curve='concave', direction='increasing')
    model = models[kn.knee]
    plt.plot([km.n_clusters for km in models], scores)
    plt.vlines(model.n_clusters, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.title(f'Knee at {model.n_clusters}')
    plt.savefig(f'{out_folder}/{label}_cluster_knee.png')
    plt.close()
    return model


def triplets(by_label, n = 50000):
    l = list(by_label.keys())
    for i in range(n):
        pos_label = np.random.randint(0, len(l))
        neg_label = np.random.randint(0, len(l))
        while neg_label == pos_label:
            neg_label = np.random.randint(0, len(l))
            
        anc_i     = np.random.randint(0, len(by_label[pos_label]))
        pos_i     = np.random.randint(0, len(by_label[pos_label]))
        neg_i     = np.random.randint(0, len(by_label[neg_label]))
        yield by_label[pos_label][anc_i], by_label[pos_label][pos_i], by_label[neg_label][neg_i]

            
def train_triplets(enc, by_label):
    model = triplet_model(WINDOW_PARAM, enc, LATENT)
    for epoch in range(EPOCHS):
        batch_pos = []
        batch_neg = []
        batch_anc = []
        n = 0
        total_loss = 0.0
        for anc, pos, neg in triplets(by_label):
            batch_pos.append(pos)
            batch_neg.append(neg)
            batch_anc.append(anc)
            if len(batch_pos) == BATCH:
                batch_anc = np.stack(batch_anc)
                batch_pos = np.stack(batch_pos)
                batch_neg = np.stack(batch_neg)
                loss = model.train_on_batch(x=[batch_anc, batch_pos, batch_neg], y=np.zeros((BATCH,  256)))

                batch_pos = []
                batch_neg = []
                batch_anc = []

                total_loss += loss
                n += 1
                if n % 10 == 0:
                    print("EPOCH: {} LOSS: {}".format(epoch, total_loss))
                    total_loss = 0.0
                    n = 0
    return model


def neighbours_encoder(encoder, x_train, y_train, x_test, y_test, label_dict, name, out_folder):
    x_train = encoder.predict(x_train, batch_size = BATCH)
    x_test = encoder.predict(x_test, batch_size = BATCH)

    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(x_train)
    index.createIndex({'post': 2}, print_progress=True)
    neighbours = index.knnQueryBatch(x_test, k=10, num_threads=4)

    n = len(label_dict)
    label_names = ["" for i in range(n)]
    for l, i in label_dict.items():
        label_names[i] = l
    confusion = np.zeros((n,n))

    for i, (ids, _) in enumerate(neighbours):
        labels = [int(y_train[i]) for i in ids]
        c      = Counter(labels)
        l      = [(k, v) for k, v in c.items()]
        l      = sorted(l, key = lambda x: x[1], reverse=True)[0][0]
        confusion[y_test[i], l] += 1

    accuracy = np.sum(confusion * np.eye(n)) / len(y_test)
    plot_result_matrix(confusion, label_names, label_names, "confusion {} {}".format(name, accuracy))
    plt.savefig('{}/confusion_nn_{}.png'.format(out_folder, name))
    plt.close()
    return accuracy

    
def train(label_file, wav_file, out_folder="output", perc_test=0.33, retrain=True, super_epochs=3, relabel=False, resample=10000):
    instances, ra, labels, label_dict = dataset_supervised_windows(
        label_file, wav_file, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, raw_size=RAW_AUDIO)    
    reverse = dict([(v, k) for k, v in label_dict.items()])
    print(label_dict)
    
    by_label = {}
    for i in range(0, len(instances)):
        y = labels[i] 
        if y not in by_label:
            by_label[y] = []            
        by_label[y].append(instances[i])
    print([(k, len(v)) for k, v in by_label.items()])
    
    if retrain:    
        _instances = []
        _labels = []
        for k, v in by_label.items():
            for _ in range(0, resample):
                i = np.random.randint(0, len(v))
                if reverse[k] != 'NOISE':
                    noise = by_label[label_dict['NOISE']]
                    ni = np.random.randint(0, len(noise))
                    _labels.append(k)
                    _instances.append((v[i] + noise[ni]) / 2.0)
                else:
                    _labels.append(k)
                    _instances.append(v[i])
                    
        y_train = []
        y_test  = []
        x_train = []
        x_test  = []
        for i in range(0, len(_instances)):                    
            if np.random.uniform() < perc_test:
                x_test.append(_instances[i])
                y_test.append(_labels[i])
            else:            
                x_train.append(_instances[i])
                y_train.append(_labels[i])

        x_train = np.stack(x_train).reshape(len(x_train), T, D, 1)
        x_test  = np.stack(x_test).reshape(len(x_test), T, D, 1)    
        y_train = np.array(y_train)
        y_test  = np.array(y_test)

        print("Train: {} / {}".format(x_train.shape, Counter(y_train)))
        print("Test:  {} / {}".format(x_test.shape, Counter(y_test)))
        
        base_encoder = encoder(WINDOW_PARAM, LATENT, CONV_PARAM)    
        base_encoder.summary()
        enc = window_encoder(WINDOW_PARAM, base_encoder, LATENT)

        accuracy_supervised    = []
        accuracy_nn_supervised = []
        accuracy_siamese       = []
        accuracy_ae            = []
        for i in range(0, super_epochs):            
            model       = classifier(WINDOW_PARAM, enc, LATENT, 5, CONV_PARAM) 
            model.summary()
            hist        = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=BATCH, epochs=EPOCHS, shuffle=True)
            
            model.save('{}/supervised.h5'.format(out_folder))
            enc.save('{}/encoder.h5'.format(out_folder))     
            base_encoder.save('{}/base_encoder.h5'.format(out_folder))
            enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_supervised.png".format(out_folder))        
            plot_tensorflow_hist(hist, "{}/history_train_supervised.png".format(out_folder))        
            acc_nn = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "classifier", out_folder)
            accuracy_nn_supervised.append(acc_nn)

            if relabel:            
                prediction = model.predict(x_train)
                y_train    = prediction.argmax(axis=1)
                
            n = len(label_dict)        
            label_names = ["" for i in range(n)]
            for l, i in label_dict.items():
                label_names[i] = l
            prediction_test = model.predict(x_test)
            confusion = np.zeros((n,n))
            correct = 0
            for i in range(len(y_test)):
                pred = np.argmax(prediction_test[i])
                confusion[y_test[i], pred] += 1
            accuracy = np.sum(confusion * np.eye(n)) / len(y_test)
            plot_result_matrix(confusion, label_names, label_names, "confusion acc {}".format(accuracy))
            plt.savefig('{}/confusion_type.png'.format(out_folder))
            plt.close()
            accuracy_supervised.append(accuracy)
            
            siamese = train_triplets(enc, by_label)
            siamese.save('{}/siam.h5'.format(out_folder))
            enc.save('{}/encoder.h5'.format(out_folder))    
            base_encoder.save('{}/base_encoder.h5'.format(out_folder))            
            enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_siam.png".format(out_folder))                
            acc_siam = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "siamese", out_folder)
            accuracy_siamese.append(acc_siam)
            
            ae          = auto_encoder(WINDOW_PARAM, enc, LATENT, CONV_PARAM)    
            ae.summary()
            hist        = ae.fit(x=x_train, y=x_train, batch_size=BATCH, epochs=EPOCHS, shuffle=True)
            ae.save('{}/ae.h5'.format(out_folder))
            enc.save('{}/encoder.h5'.format(out_folder))        
            base_encoder.save('{}/base_encoder.h5'.format(out_folder))
            enc_filters(enc, N_FILTERS, N_BANKS, "{}/filters_ae.png".format(out_folder))                
            plot_tensorflow_hist(hist, "{}/history_train_ae.png".format(out_folder))
            visualize_dataset(ae.predict(x_test, batch_size=BATCH), "{}/reconstructions.png".format(out_folder))
            acc_ae = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "aute encoder", out_folder)
            accuracy_ae.append(acc_ae)
            enc.save('{}/encoder.h5'.format(out_folder))                
            pkl.dump(label_dict, open('{}/labels.pkl'.format(out_folder), "wb"))

        plt.plot(accuracy_supervised, label="supervised")
        plt.plot(accuracy_nn_supervised, label="nn_supervised")
        plt.plot(accuracy_siamese, label="nn_siam")
        plt.plot(accuracy_ae, label="nn_ae")
        plt.legend()
        plt.title("Super Epochs")
        plt.savefig('{}/super_epoch_acc.png'.format(out_folder))
        plt.close()
    else:
        model = load_model('{}/supervised.h5'.format(out_folder))
        enc   = load_model('{}/encoder.h5'.format(out_folder))    
      
    by_label = dict([(k, enc.predict(np.stack(v), batch_size=10)) for k, v in by_label.items()])
    clusters = dict([(k, cluster_model(v, out_folder, reverse[k])) for k, v in by_label.items() if k != label_dict['NOISE']])
    pkl.dump(clusters, open('{}/clusters_window.pkl'.format(out_folder),'wb'))
    print(f'Done Clustering: {[(k, v.cluster_centers_.shape) for k, v in clusters.items()]}')
    
    b = np.stack(instances)
    h = enc.predict(b, batch_size=10)
    x = model.predict(b, batch_size=10)
    extracted = {}
    for n, i in enumerate(x):
        if n % 1000 == 0:
            print(f"{n} of {len(x)}")
        li = int(np.argmax(i))
        l = reverse[li]
        if l != 'NOISE':
            hn      = h[n].reshape(1, LATENT)
            pred_hn = clusters[li].predict(hn)
            c = int(pred_hn[0])    
            if l not in extracted:
                extracted[l] = {}
            if c not in extracted[l]:
                extracted[l][c] = []
            if li != labels[n]:
                l_true = reverse[labels[n]]
                if l_true not in extracted[l]:
                    extracted[l][l_true] = []
                extracted[l][l_true].append(ra[n])
            extracted[l][c].append(ra[n])

    for l, clusters in extracted.items():
        for c, audio in clusters.items():
            path = "{}/{}_{}.wav".format(out_folder, l, c)
            write(path, 44100, np.concatenate(audio))


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def train_sequential(folder, labels, data, noise):
    ids         = pkl.load(open(f"{folder}/ids.pkl", "rb"))
    inst        = pkl.load(open(f"{folder}/instances.pkl", "rb"))
    predictions = [x for x in pkl.load(open(f"{folder}/predictions.pkl", "rb"))]
    lab         = pkl.load(open(f"{folder}/labels.pkl", "rb"))
    
    df      = pd.read_csv(labels)
    signals = raw(data)
    noise   = raw(noise)

    reverse = {v:k for k, v in lab.items()}
    
    ranges = []
    for _, row in df.iterrows():
        ranges.append([row['starts'], row['stops']])

    encoder       = load_model(f'{folder}/base_encoder.h5')
    clst          = pkl.load(open(f"{folder}/clusters_window.pkl", "rb"))
    label_mapping = LabelMapping.mapping(clst)
    pkl.dump(label_mapping, open(f'{folder}/label_mapping.pkl', 'wb'))

    dim = np.sum([c.n_clusters for c in clst.values()]) + 1
    opt = SGD(learning_rate=0.01, momentum=0.9)
    decoder = seq2seq_classifier(WINDOW_PARAM, encoder, LATENT, dim)
    decoder.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    decoder.summary()
    
    TOTAL = len(predictions)
    accuracies = []
    for i in range(0, TOTAL * 25):
        if i % 100 == 0 and i > 0:
            print(f'Epoch: {i} {np.mean(accuracies[-100:])} ')
        batch_x, batch_y, y = get_batch(signals, noise, inst, ranges, ids, predictions, dim, clst, label_mapping,\
                                        FFT_LO, FFT_HI, FFT_WIN, FFT_STEP, T, batch = 3)    
        loss, acc = decoder.train_on_batch(x=batch_x, y=batch_y)
        accuracies.append(acc)
    decoder.save(f'{folder}/decoder_nn.h5')
    enc_filters(encoder, N_FILTERS, N_BANKS, f'{folder}/decoder_nn_filters.png')
    accuracies = np.convolve(accuracies, np.ones(TOTAL), 'valid') / TOTAL
    plt.plot(moving_average(accuracies, TOTAL))
    plt.xlabel('iter')
    plt.ylabel('acc')
    plt.savefig(f'{folder}/acc_seq2seq.png')
    plt.close()
    
    
def clustering(regions, wav_file, folder, l2_window = None): # 10):
    instances_file   = "{}/instances.pkl".format(folder)
    ids_file         = "{}/ids.pkl".format(folder)
    predictions_file = "{}/predictions.pkl".format(folder)
    distances_file   = "{}/distances.pkl".format(folder)
    clusters_file    = "{}/clusters.pkl".format(folder)

    label_dict = pkl.load(open("{}/labels.pkl".format(folder), "rb"))
    reverse = dict([(v,k) for k, v in label_dict.items()])
    if not os.path.exists(instances_file):
        cls = load_model('{}/supervised.h5'.format(folder))
        enc = load_model('{}/encoder.h5'.format(folder))
        if l2_window is not None:
            ids, instances, predictions = dataset_unsupervised_regions_windowed(
                regions, wav_file, enc, cls, reverse, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, T=T, l2_window=l2_window, dont_window_whistle=True)
        else:
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


def export(csvfile, wavfile, folder, k, out, prefix, min_c = 2):
    print(" ... loading data")
    
    label_file       = "{}/labels.pkl".format(folder)
    ids_file         = "{}/{}ids.pkl".format(folder, prefix)
    predictions_file = "{}/{}predictions.pkl".format(folder, prefix)
    clusters_file    = "{}/{}clusters.pkl".format(folder, prefix)

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

    
def htk_train(folder, inputs, states, niter, k, flat=False):
    print("Prepare project: {}".format(folder))
    out = check_output(["rm", "-rf", folder])
    out = check_output(["mkdir", folder])
    out = check_output(["mkdir", "{}/data".format(folder)])
    htk_export(inputs, "{}/data".format(folder), "{}/clusters.mlf".format(folder), folder, k)
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
        print("INIT")
        htk_init("{}/clusters_TRAIN.mlf".format(folder), None, LATENT, "{}/data/train/*.htk".format(folder), folder, LATENT, states, "{}/hmm0".format(folder), "{}/list".format(folder))
    out = check_output("HParse {}/gram {}/wdnet".format(folder, folder).split(" "))
    
    print("---")
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
    conf, names = htk_confusion("{}/predictions.mlf".format(folder))
    plot_result_matrix(conf, names, names, "Confusion Window")
    plt.savefig("{}/confusion_window.png".format(folder))
    plt.close()
    

def htk_converter(file, folder, out):
    print("... convert {} using {} to {}".format(file, folder, out))
    enc      = load_model('{}/encoder.h5'.format(folder))
    audio    = raw(file) 
    spec     = spectrogram(audio, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
    windowed = windowing(spec, T)
    x        = enc.predict(windowed, batch_size=10)
    return write_htk(x, out), x, windowed


def htk_continuous(folder, htk, noise, hmm, components=10):
    htk_file = "{}/data/{}".format(htk, noise.split('/')[-1].replace('.wav', '.htk'))
    n,x,_    = htk_converter(noise, folder, htk_file)
    out      = check_output(["rm", "-rf", "{}/sil0".format(htk)])
    out      = check_output(["mkdir", "{}/sil0".format(htk)])

    km = KMeans(components)
    km.fit(x)
    cmp = km.cluster_centers_

    with open("{}/sil0/sil".format(htk), "w") as fp:
        model = silence_proto(LATENT, cmp)
        fp.write(model)    

    with open("{}/list_sil".format(htk), "w") as fp:
        fp.write("sil\n")

    with open("{}/clusters_sil.mlf".format(htk), "w") as fp:
        fp.write("#!MLF!#\n")
        fp.write("\"*/{}\"\n".format(noise.split('/')[-1].replace('.wav', '.lab')))
        fp.write("{} {} sil\n".format(0, n))
        fp.write(".\n")
    
    out = check_output("HERest -A -T 1 -v {} -I {}/clusters_sil.mlf -M {}/sil0 -H {}/sil0/sil {}/list_sil".format(FLOOR, htk, htk, htk, htk).split(" ") + [htk_file])
    print("Sil LL: {}".format(get_ll(out)))

    out = check_output("cp {} {}/continuous".format(hmm, htk).split(" "))
    with open("{}/continuous".format(htk), "a") as fp:
        for i, line in enumerate(open("{}/sil0/sil".format(htk))):
            if i > 2:
                fp.write(line)

    out = check_output("cp {}/list {}/list_continuous".format(htk, htk).split(" "))
    with open("{}/list_continuous".format(htk), "a") as fp:
        fp.write("\nsil\n")

    grammar = simple_grammar("{}/clusters_TRAIN.mlf".format(htk), True)
    with open("{}/gram_continuous".format(htk), 'w') as fp:
        fp.write(grammar + "\n")

    wlist = wordlist("{}/clusters_TRAIN.mlf".format(htk), True)
    with open("{}/dict_continuous".format(htk), 'w') as fp:
        fp.write(wlist + "\n")
    
    out = check_output("HParse {}/gram_continuous {}/wdnet_continuous".format(htk, htk).split(" "))
                

def sequencing(audio, folder, htk, outfolder, recode=False):
    print("SEQUENCING")
    if recode:        
        out = check_output(["rm", "-rf", outfolder])
        out = check_output(["mkdir", outfolder])
        out = check_output(["mkdir", "{}/images".format(outfolder)]) 

        model      = load_model('{}/supervised.h5'.format(folder))
        label_dict = pkl.load(open('{}/labels.pkl'.format(folder), "rb"))

        n = len(label_dict)
        label_names = ["" for i in range(n)]
        for l, i in label_dict.items():
            label_names[i] = l    

        htk_files   = []
        label_files = []
        for file in os.listdir(audio):
            if file.endswith(".wav"):
                path         = "{}/{}".format(audio, file)
                out_path     = "{}/{}".format(outfolder, file).replace(".wav", ".htk")
                out_path_lab = "{}/{}".format(outfolder, file).replace(".wav", ".csv")

                _, _, w = htk_converter(path, folder, out_path)

                y = model.predict(w)            
                p = [np.max(y[i]) for i in range(len(y))]
                l = [np.argmax(y[i]) for i in range(len(y))]
                
                y = [label_names[i]  for i in l] 
                
                df = pd.DataFrame({
                    'labels': y,
                    'prob': p
                })
                df.to_csv(out_path_lab, index=False)
                htk_files.append(out_path)
                label_files.append(out_path_lab)
                print("Convert: {}".format(path))

        # TODO Adjust silence model
        cmd = "HVite -H {}/continuous -i {}/sequenced.lab -w {}/wdnet_continuous {}/dict_continuous {}/list_continuous"\
            .format(htk, outfolder, htk, htk, htk)\
            .split(' ')
        cmd.extend(htk_files)
        out = check_output(cmd)    
    
    annotations         = parse_mlf('{}/sequenced.lab'.format(outfolder))
    th                  = htk_threshold('{}/sequenced.lab'.format(outfolder), outfolder)        
    label_files         = dict([(f.replace('.csv', ''), "{}/{}".format(outfolder, f))for f in os.listdir(outfolder) if f.endswith('.csv')])
    filtered            = plot_annotations(annotations, label_files, audio, "{}/images".format(outfolder), T // 2, th)
    htk_sequencing_eval(filtered, outfolder)


def discrete_clustering(folder, regions, wav_file):
    instances_file   = "{}/discrete_instances.pkl".format(folder)
    ids_file         = "{}/discrete_ids.pkl".format(folder)
    predictions_file = "{}/discrete_predictions.pkl".format(folder)
    string_file      = "{}/discrete_strings.pkl".format(folder)
    distances_file    = "{}/discrete_distances.pkl".format(folder)
    clusters_file    = "{}/discrete_clusters.pkl".format(folder)
    
    label_dict = pkl.load(open("{}/labels.pkl".format(folder), "rb"))
    reverse = dict([(v,k) for k, v in label_dict.items()])
    
    if not os.path.exists(instances_file):
        sub = load_model('{}/supervised.h5'.format(folder))
        enc = load_model('{}/encoder.h5'.format(folder))
        ids, instances, predictions = dataset_unsupervised_regions(
            regions, wav_file, enc, sub, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, T=T)   
        print("#Instances: {}".format(len(instances)))
        pkl.dump(ids, open(ids_file, "wb"))
        pkl.dump(instances, open(instances_file, "wb"))
        pkl.dump(predictions, open(predictions_file, "wb"))
    else:
        instances   = pkl.load(open(instances_file, "rb"))
        predictions = pkl.load(open(predictions_file, "rb"))

    if not os.path.exists(string_file):
        clu = pkl.load(open('{}/clusters_window.pkl'.format(folder),'rb'))    
        strings = []
        for i in range(0, len(predictions)):
            label = predictions[i].argmax(axis = 1)    
            strg  = [clu[l].predict(instances[i][j].reshape(1, LATENT))[0] for j, l in enumerate(label)]
            symb  = symbols(strg, label) 
            strings.append(symb)
        pkl.dump(strings, open(string_file, "wb"))
    else:
        strings = pkl.load(open(string_file, "rb"))
        
    if not os.path.exists(distances_file):        
        distances = levenstein_distances(strings)
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

    
def discrete_decoding(folder, audio, out_folder):
    SCALER = 1.0
    BIAS   = 0.7
    START  = 0.2
    STOP   = 0.9

    DAMPEN_NOISE = 0.01
    
    sub = load_model('{}/supervised.h5'.format(folder))
    enc = load_model('{}/encoder.h5'.format(folder))
    clu = pkl.load(open('{}/clusters_window.pkl'.format(folder),'rb'))
    label_dict = pkl.load(open("{}/labels.pkl".format(folder), "rb"))
    reverse = dict([(v,k) for k, v in label_dict.items()])

    out = check_output(["rm", "-rf", out_folder])
    out = check_output(["mkdir", out_folder])

    cluster_labels = {}
    cur = 0
    by_file = {}
    for filename in os.listdir(audio):
        if filename.endswith('.wav'):
            path        = "{}/{}".format(audio, filename)
            img_name    = filename.replace('.wav', '.png')
            a           = spectrogram(raw(path), lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP)
            
            if len(a) < 50000:
                windowed = windowing(a, T)
                p        = sub.predict(windowed)

                p[:, label_dict['NOISE']] *= DAMPEN_NOISE
                
                e        = enc.predict(windowed)     
                ay       = np.argmax(p, axis = 1)

                fig, ax = plt.subplots()
                fig.set_size_inches(len(a) / 100, len(a[0]) / 100)
                ax.imshow(BIAS - a.T * SCALER, norm=Normalize(START, STOP), cmap='gray')                  
                annotations = []
                strg = []
                for i, y in enumerate(ay):       
                    c = clu[y].predict(e[i].reshape(1, LATENT))[0]
                    lab = reverse[y]
                    v = ' '
                    if lab == 'WSTL_DOWN':
                        v = 'D' 
                    if lab == 'WSTL_UP':
                        v = 'U'
                    if lab == 'ECHO':
                        v = 'E'
                    if lab == 'BURST':
                        v = 'B'
                    s = "{}{}".format(v, c)
                    if lab != 'NOISE':
                        strg.append(s)
                        annotations.append([i * T // 2, (i + 1) * T // 2, s, 1.0])
                if len(annotations) > 0:    
                    anno = compress(annotations)
                    anno = [(start, stop, s) for start, stop, s, _ in anno]

                    for start, stop, s in anno:
                        if s not in cluster_labels:
                            cluster_labels[s] = cur
                            cur += 1
                        color =  cluster_labels[s]
                        plt.text(start + (stop - start) // 2 - 7 , 30, s[0], size=10, color='black')
                        rect = patches.Rectangle((start, 0), stop - start,
                                                 256, linewidth=1, edgecolor='r', facecolor=COLORS[color])
                        ax.add_patch(rect)
                    path = '{}/{}'.format(out_folder, img_name)
                    plt.savefig(path)
                    plt.close()
                    by_file[filename] = (anno, strg, img_name)
                
    with open('{}/sequenced_strings.html'.format(out_folder), 'w') as f:
        f.write('<HTML><BODY><TABLE border="1">')
        f.write("""
        <TR>
            <TH> Context </TH>
            <TH> Video </TH>
            <TH> Time </TH>
            <TH> String </TH>
            <TH> Image </TH>
        </TR>    
        """)
        for file, (_, strg, p) in by_file.items():
            f.write("""
            <TR>
                <TD> {} </TD>
                <TD> {} </TD>
                <TD> {} </TD>
                <TD> {} </TD>
                <TD> 
                   <div style="width: 1024px; height: 100px; overflow: auto">
                     <img src="{}" height=100/> </div></TD>
            </TR>    
            """.format(
                context(file), 
                video(file, context(file)), 
                timestamp(file), 
                " ".join(strg),
                p
            ))
        f.write('</TABLE></BODY></HTML>')        


def i2name(i, reverse, label_mapping):    
    if i == 0:
        return 'NOISE'
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
    

def neural_decoding(folder, in_folder, out_folder):
    decoder = load_model(f'{folder}/decoder_nn.h5')
    lab     = pkl.load(open(f"{folder}/labels.pkl", "rb"))
    reverse = {v:k for k, v in lab.items()}
    label_mapping = pkl.load(open(f'{folder}/label_mapping.pkl', 'rb'))

    images = []
    strings = []
    for f in os.listdir(in_folder):
        if f.endswith('.wav'):        
            x = raw(f'{in_folder}/{f}')
            s = spectrogram(x, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
            if len(s) < 100000:
                c = []
                for i in range(0, len(s), 1000):
                    x = s[i:i + 1000]
                    a = x.reshape((1, len(x), D, 1))
                    p = decoder.predict(a).reshape((a.shape[1], label_mapping.n + 1)) 
                    if len(p) > NEURAL_SMOOTH_WIN:
                        for i in range(0, len(p[0])):
                            p[:, i] = np.convolve(p[:, i], np.ones(NEURAL_SMOOTH_WIN) / NEURAL_SMOOTH_WIN, mode='same')
                    p[:, 0] *= NEURAL_NOISE_DAMPENING
                    local_c = p.argmax(axis=1)
                    c += list(local_c)
                if len([l for l in c if l > 0]) > 3:
                    fig, ax = plt.subplots()
                    fig.set_size_inches(len(s) / 100, len(s[0]) / 100)
                    ax.imshow(1.0 - s.T,  cmap='gray')                  
                    last = 0
                    strg = []
                    for i in range(1, len(c)):
                        if c[i] != c[i - 1]:                       
                            if c[i - 1] != 0:  
                                start = last
                                stop = i
                                if stop - start > NEURAL_SIZE_TH:
                                    strg.append(c[i - 1])
                                    rect = patches.Rectangle((start, 0), stop - start,
                                                             256, linewidth=1, edgecolor='r', facecolor=COLORS[c[i - 1]])
                                    ax.add_patch(rect)
                                    plt.text(start + (stop - start) // 2 , 30, i2name(c[i - 1], reverse, label_mapping), size=12)
                            last = i
                    if last != len(s) and c[-1] != 0:
                        strg.append(c[i - 1])
                        i = len(s) - 1
                        start = last
                        stop = i
                        rect = patches.Rectangle((start, 0), stop - start,
                                         256, linewidth=1, edgecolor='r', facecolor=COLORS[c[i - 1]])
                        ax.add_patch(rect)
                        plt.text(start + (stop - start) // 2 , 30, i2name(c[i - 1], reverse, label_mapping), size=12)
                    p = f.replace('.wav', '.png')
                    img_path = f'{out_folder}/{p}'
                    plt.savefig(img_path)
                    plt.close()
                    strings.append(strg)
                    images.append(p)

    N = len(strings)
    d = np.zeros((N, N))
    for i in range(0, N):
        for j in range(i, N):
            l = levenstein(strings[i], strings[j])
            d[i, j] = l
            d[j, i] = l

    j, di = merge_next(0, d, set([]))
    closed  = set([j]) 

    seq_sorted = []
    img_sorted = []
    while di < np.float('inf'):
        j, di = merge_next(j, d, closed)
        closed.add(j)
        seq_sorted.append(" ".join([i2name(s, reverse, label_mapping) for s in strings[j]]))
        img_sorted.append(images[j])
        
    with open(f'{out_folder}/sequenced_strings.html', 'w') as f:
        f.write('<HTML><BODY><TABLE border="1">')
        f.write("""
        <TR>
            <TH> String </TH>
            <TH> Image </TH>
        </TR>    
        """)
        for seq, img in zip(seq_sorted, img_sorted):
            img = "/".join(img.split('/')[-2:])
            f.write("""
            <TR>
                <TD> {} </TD>
                <TD> 
                   <div style="width: 1024px; height: 100px; overflow: auto">
                     <img src="{}" height=100/> </div></TD>
            </TR>    
            """.format(
                seq, img
            ))
        f.write('</TABLE></BODY> </HTML>')

        
def join_wav(folder, out_wav, out_csv):
    raw_file = []
    offest = [] 
    starts = []
    stops  = []
    total = 0
    for file in os.listdir(folder):        
        if file.endswith('.wav'):
            path = "{}/{}".format(folder, file)
            x = raw(path)
            raw_file.append(x)
            starts.append(total)
            total += len(x)
            stops.append(total)
    raw_file = np.hstack(raw_file)   
    df = pd.DataFrame({
        'starts': starts,
        'stops': stops
    })
    df.to_csv(out_csv)
    write(out_wav, 44100, raw_file)

    
def neardup(query_folder, labels, wav, folder, out, k = 10, percentile=50, band=0.01, max_len_diff=5):    
    ids         = pkl.load(open(f"{folder}/ids.pkl", "rb"))
    inst        = pkl.load(open(f"{folder}/instances.pkl", "rb"))
    d           = pkl.load(open(f"{folder}/distances.pkl", "rb")) 
    df      = pd.read_csv(labels)
    signals = raw(wav)
    
    th = np.percentile(d.flatten(), percentile)
    print(f"Threshold: {th}")

    ranges = []
    for _, row in df.iterrows():
        ranges.append([row['starts'], row['stops']])

    encoder = load_model(f'{folder}/encoder.h5')
    
    names = []
    queries = []
    for f in os.listdir(query_folder):
        if f.endswith('.wav'):            
            path = "{}/{}".format(query_folder, f)
            x = raw(path)
            if len(x) > 0:            
                s = spectrogram(x, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
                w = windowing(s, T)
                e = encoder.predict(w)
                queries.append(e)
                names.append(f)

    distances = np.ones((len(queries), len(inst))) * float('inf')
    for i, q in enumerate(queries):
        for j, x in enumerate(inst):
            if np.abs(len(x) - len(q)) < max_len_diff:
                d = dtw(q, x, band)
                distances[i, j] = d
            
    for i in range(len(queries)):
        audio = []
        out_wav = f"{out}/{names[i]}"
        neighbors = [(j, d) for j, d in enumerate(distances[i])]
        neighbors = sorted(neighbors, key=lambda x: x[1])
        neighbors = [(j,d) for j, d in neighbors if d < th] 
        for j, _ in neighbors[0:k]:
            start, stop = ranges[ids[j]]
            audio.append(signals[start:stop])
        audio = np.hstack(audio)   
        write(out_wav, 44100, audio)
        print(neighbors[0:k])
        
    
if __name__ == '__main__':
    print("=====================================")
    print("Simplified WDP DS Pipeline")
    print("by Daniel Kyu Hwa Kohlsdorf")
    if len(sys.argv) >= 5 and sys.argv[1] == 'train':            
        labels = sys.argv[2]
        wav    = sys.argv[3]
        out    = sys.argv[4]        
        train(labels, wav, out)
    elif len(sys.argv) >= 5 and sys.argv[1] == 'join':
        folder  = sys.argv[2]
        wav_out = sys.argv[3]
        csv_out = sys.argv[4]
        join_wav(folder, wav_out, csv_out)
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
        prefix   = sys.argv[6]
        out      = sys.argv[7]
        export(labels, wav, clusters, k, out, prefix)
    elif len(sys.argv) >= 6 and sys.argv[1] == 'htk':
        mode   = sys.argv[2]
        if mode == 'train':
            inputs = sys.argv[3]
            folder = sys.argv[4]
            states = int(sys.argv[5])
            niter  = int(sys.argv[6])
            k      = int(sys.argv[7])
            htk_train(folder, inputs, states, niter, k)
        elif mode == 'continuous':
            folder = sys.argv[3]
            htk    = sys.argv[4]
            noise  = sys.argv[5]
            hmm    = sys.argv[6]
            htk_continuous(folder, htk, noise, hmm)
        else:
            audio  = sys.argv[3]
            folder = sys.argv[4]
            htk    = sys.argv[5]
            htk_file = "{}/{}".format(htk, audio.split('/')[-1].replace('.wav', '.htk'))
            htk_converter(audio, folder, htk_file)
    elif len(sys.argv) >= 3 and sys.argv[1] == 'baseline':
        folder = sys.argv[2]
        dtw_baseline(folder)
    elif len(sys.argv) > 5 and sys.argv[1] == 'sequencing':
        audio  = sys.argv[2]
        folder = sys.argv[3]
        htk    = sys.argv[4]
        out    = sys.argv[5]
        sequencing(audio, folder, htk ,out)
    elif len(sys.argv) > 5 and sys.argv[1] == 'discrete':
        if sys.argv[2] == 'clustering':
            labels = sys.argv[3]
            wav    = sys.argv[4]
            out    = sys.argv[5]
            discrete_clustering(out, labels, wav)
        if sys.argv[2] == 'sequencing':
            audio  = sys.argv[3]
            folder = sys.argv[4]
            out    = sys.argv[5]
            discrete_decoding(folder, audio, out)
    elif len(sys.argv) > 6 and sys.argv[1] == 'neardup':
        query_folder = sys.argv[2]
        labels = sys.argv[3]        
        wav    = sys.argv[4]
        folder = sys.argv[5]
        out    = sys.argv[6]
        neardup(query_folder, labels, wav, folder, out)
    elif len(sys.argv) > 5 and sys.argv[1] == 'train_sequential':
        folder = sys.argv[2]
        labels = sys.argv[3]        
        data   = sys.argv[4]
        noise  = sys.argv[5]        
        train_sequential(folder, labels, data, noise)
    elif len(sys.argv) > 4 and sys.argv[1] == 'decode_neural':
        folder = sys.argv[2]
        in_folder = sys.argv[3]
        out_folder = sys.argv[4]
        neural_decoding(folder, in_folder, out_folder)
    else:
        print(sys.argv)
        print("""
            Usage:
                + train:      python pipeline.py train LABEL_FILE AUDIO_FILE OUT_FOLDER
                + seq2seq:    python pipeline.py train_sequential FOLDER LAB WAV NOISE
                              python pipeline.py decode_neural FOLDER IN OUT
                + nearest:    python pipeline.py neardup QUERY_FOLDER LAB WAV FOLDER OUT_FOLDER
                + join:       python pipeline.py join FOLDER_2_JOIN WAV_OUT CSV_OUT
                + clustering: python pipeline.py clustering LABEL_FILE AUDIO_FILE OUT_FOLDER
                + export:     python pipeline.py export LABEL_FILE AUDIO_FILE FOLDER K PREFIX OUT_FOLDER
                + discrete    python pipeline.py discrete clustering LABEL_FILE AUDIO_FILE OUT_FOLDER
                              python pipeline.py discrete sequencing AUDIO_FOLDER FOLDER OUT_FOLDER
                + htk:        python pipeline.py htk train FOLDER OUT_HTK STATES ITER K
                              python pipeline.py htk continuous FOLDER OUT_HTK NOISE HMM
                              python pipeline.py htk convert AUDIO FOLDER OUT_FOLDER 
                + sequencing: python pipeline.py sequencing AUDIO FOLDER HTK OUT
                + baseline:   python pipeline.py baseline FOLDER
        """)
    print("\n=====================================")
