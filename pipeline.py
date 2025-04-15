import numpy as np
import pickle as pkl
import sys
import os

import nmslib

from lib_dolphin.audio import *
from lib_dolphin.features import *
from lib_dolphin.parameters import *
from lib_dolphin.discovery import *
from lib_dolphin.connected_components import *

from collections import namedtuple, Counter, defaultdict

from scipy.io.wavfile import read, write
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *

from subprocess import check_output


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
    x_train = encoder.predict(x_train, batch_size = BATCH, verbose = 0)
    x_test = encoder.predict(x_test, batch_size = BATCH, verbose = 0)

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
    return accuracy


def group_by_label(instances, labels):
    by_label = {}
    for i in range(0, len(instances)):
        y = labels[i]
        if y not in by_label:
            by_label[y] = []
        by_label[y].append(instances[i])
    return by_label


def add_noise(by_label, label_dict, reverse, resample):
    instances = []
    labels = []
    for k, v in by_label.items():
        for _ in range(0, resample):
            i = np.random.randint(0, len(v))
            if reverse[k] != 'NOISE':
                noise = by_label[label_dict['NOISE']]
                ni = np.random.randint(0, len(noise))
                labels.append(k)
                instances.append((v[i] + noise[ni]) / 2.0)
            else:
                labels.append(k)
                instances.append(v[i])
    return instances, labels


def split_train_test(instances, labels, perc_test):
    y_train = []
    y_test  = []
    x_train = []
    x_test  = []
    for i in range(0, len(instances)):
        if np.random.uniform() < perc_test:
            x_test.append(instances[i])
            y_test.append(labels[i])
        else:
            x_train.append(instances[i])
            y_train.append(labels[i])

    x_train = np.stack(x_train).reshape(len(x_train), T, D, 1)
    x_test  = np.stack(x_test).reshape(len(x_test), T, D, 1)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)
    return x_train, y_train, x_test, y_test


def train(label_file, wav_file, label_file_l2, wav_file_l2, out_folder="output", perc_test=0.33, super_epochs=3, resample=10000):
    instances, ra, labels, label_dict = dataset_supervised_windows(
        label_file, wav_file, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, raw_size=RAW_AUDIO)

    x_unsupervised = dataset_unsupervised_windows(label_file_l2, wav_file_l2, lo=FFT_LO, hi=FFT_HI, win=FFT_WIN, step=FFT_STEP, raw_size=RAW_AUDIO, T=T, n=10000)
    x_unsupervised = np.stack(x_unsupervised).reshape(len(x_unsupervised), T, D, 1)

    reverse = dict([(v, k) for k, v in label_dict.items()])
    by_label = group_by_label(instances, labels)
    pkl.dump(label_dict, open('{}/labels.pkl'.format(out_folder), "wb"))

    instances_noise, labels_noise =  add_noise(by_label, label_dict, reverse, resample)
    x_train, y_train, x_test, y_test = split_train_test(instances_noise, labels_noise, perc_test)

    print("================= LABLES ====================")
    print(label_dict)
    print([(k, len(v)) for k, v in by_label.items()])
    print("=============================================")

    print(f"Unsupervised: {x_unsupervised.shape}")
    print(f"Train: {x_train.shape} / {Counter(y_train)}")
    print(f"Test:  {x_test.shape} / {Counter(y_test)}")

    base_encoder = encoder(WINDOW_PARAM, LATENT, CONV_PARAM)
    base_encoder.summary()
    enc = window_encoder(WINDOW_PARAM, base_encoder, LATENT)

    accuracy_supervised    = []
    accuracy_siamese       = []
    accuracy_ae            = []
    for i in range(0, super_epochs):
        siamese = train_triplets(enc, by_label)
        siamese.save('{}/siam.h5'.format(out_folder))
        enc.save('{}/encoder.h5'.format(out_folder))
        base_encoder.save('{}/base_encoder.h5'.format(out_folder))
        acc_siam = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "siamese", out_folder)
        accuracy_siamese.append(acc_siam)

        ae = auto_encoder(WINDOW_PARAM, enc, LATENT, CONV_PARAM)
        ae.summary()
        ae.fit(x=x_unsupervised, y=x_unsupervised, batch_size=BATCH, epochs=EPOCHS, shuffle=True)
        ae.fit(x=x_train, y=x_train, batch_size=BATCH, epochs=EPOCHS, shuffle=True)
        ae.save('{}/ae.h5'.format(out_folder))
        enc.save('{}/encoder.h5'.format(out_folder))
        base_encoder.save('{}/base_encoder.h5'.format(out_folder))
        acc_ae = neighbours_encoder(enc, x_train, y_train, x_test, y_test, label_dict, "auto encoder", out_folder)
        accuracy_ae.append(acc_ae)
        enc.save('{}/encoder.h5'.format(out_folder))

        model = classifier(WINDOW_PARAM, enc, c5)
        model.summary()
        model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=BATCH, epochs=EPOCHS, shuffle=True)
        n = len(label_dict)
        prediction_test = model.predict(x_test, verbose=0)
        confusion = np.zeros((n,n))
        for i in range(len(y_test)):
            pred = np.argmax(prediction_test[i])
            confusion[y_test[i], pred] += 1
        accuracy = np.sum(confusion * np.eye(n)) / len(y_test)
        accuracy_supervised.append(accuracy)
        model.save('{}/supervised.h5'.format(out_folder))
        enc.save('{}/encoder.h5'.format(out_folder))
        base_encoder.save('{}/base_encoder.h5'.format(out_folder))

    plt.plot(accuracy_supervised, label="supervised")
    plt.plot(accuracy_siamese, label="nn_siam")
    plt.plot(accuracy_ae, label="nn_ae")
    plt.legend()
    plt.title("Super Epochs")
    plt.savefig('{}/super_epoch_acc.png'.format(out_folder))
    plt.close()


def decode(classifier_path, audio_path, label_path, out_csv, batch_size=100):
    classifier = load_model(classifier_path)
    labels = pkl.load(open(label_path, 'rb'))
    n_labels = int(max(labels.values())) + 1

    audio = raw(audio_path)
    row_names = {v:k for k, v in labels.items()}
    row_names[-1] = 'sample'
    data = {k:[] for k, v in row_names.items()}

    for start in range(0, len(audio), 100000000):
        spec = spectrogram(audio[start:start+100000000], FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)
        spec = spec[0:-(len(spec) % 36), :]
        windows = spec.reshape((len(spec) // 36, 36, 130, 1))
        if len(windows) > 0:
            predictions = classifier.predict(windows, batch_size=batch_size)
            for t, row in enumerate(predictions):
                for i in range(0, n_labels):
                    data[i].append(row[i])
                data[-1].append(start + t * 36 * FFT_STEP)
                
    df = pd.DataFrame({row_names[k] : v for k, v in data.items()})    
    df['WSTL'] = df.apply(lambda x: max(x.WSTL_UP, x.WSTL_DOWN), axis=1)
    df['WSTL_REGION'] = connected_components(df['WSTL'], DETECTION_TH, SMOOTH_WIN, MIN_REGION_SZE)
    df['BURST_REGION'] = connected_components(df['BURST'], DETECTION_TH, SMOOTH_WIN, MIN_REGION_SZE)
    df['ECHO_REGION'] = connected_components(df['ECHO'], DETECTION_TH, SMOOTH_WIN, MIN_REGION_SZE)
    df.to_csv(out_csv, index=None)


def extract(audio_path, csv_path, region_col, output_folder, offset,
            min_samples = 36 * FFT_STEP, max_samples = 36 * FFT_STEP * 100):
    col = region_col.replace('_REGION', '')
    audio = raw(audio_path)
    df = pd.read_csv(csv_path)
    start = df.groupby(region_col)['sample'].min()
    start = start.reset_index()
    stop = df.groupby(region_col)['sample'].max() + (38 * FFT_STEP)
    stop = stop.reset_index()

    noise  = df.groupby(region_col)['NOISE'].prod().reset_index()
    signal = df.groupby(region_col)[col].prod().reset_index()
    
    ranges = start.merge(stop, on=region_col, suffixes=('_min', '_max'))
    ranges = ranges.merge(signal, on=region_col)
    ranges = ranges.merge(noise, on=region_col)
    ranges = ranges.fillna(0.0)
    instance_id = 0
    for i, row in ranges.iterrows():
        instance_id = offset + i
        region = audio[int(row.sample_min - FFT_WIN * 36):int(row.sample_max)]
        n_samples = len(region)        
        snr = int(row[col] /(1e-12 + row.NOISE)) // 10000
        if n_samples > min_samples and n_samples < max_samples and snr > 1:
            write(f"{output_folder}/{region_col}_{instance_id}_{snr}.wav", 44100, region)
        else:
            reason = " too short " if n_samples <= min_samples else " too long"
            print(f"\t\t REJECT: {audio_path} range {row.sample_min}:{row.sample_max} |{n_samples}| {reason}")
    return instance_id


def aligned(encoder_path, l2_labels, l2_wav, out_folder, epochs=5, batch_size=100):
    encoder = load_model(encoder_path)
    encoder.summary()
    instances = dataset_unsupervised(l2_labels, l2_wav,
                                     lo=FFT_LO, hi=FFT_HI, win=FFT_WIN,
                                     step=FFT_STEP, raw_size=RAW_AUDIO, T=T)
                                    
    n_instances = len(instances)
    for epoch in range(0, epochs):
        print(f"decoding #instaces: {n_instances} epoch: {epoch}")
        embeddings = []
        raw_windows = []
        instance_id = 0
        for spec in instances:
            if instance_id % 100 == 0:
                percentage = instance_id / n_instances
                print(f"Percentage: {percentage * 100}")
            spec = spec[0:-(len(spec) % 36), :]
            if len(spec) > 0: 
                windows = spec.reshape((len(spec) // 36, 36, 130, 1))            
                embedded_windows = encoder.predict(windows, batch_size = batch_size, verbose = 0)
                raw_windows.append(windows)
                embeddings.append(embedded_windows)
            instance_id += 1

        print("Compute distances")
        distances = pairwise_dtw_distance_matrix(embeddings)
        
        print("Clustering")
        labels = hierarchical_clustering(distances, th=np.percentile(distances, 50))

        print("Barycentering")
        groups = defaultdict(list)
        instance_ids = defaultdict(list)
        for i, (label, embedding) in enumerate(zip(labels, embeddings)):
            groups[label].append(embedding)
            instance_ids[label].append(i)    
        
        bary_centers = {
            label: dtw_barycenter_avg(sequences)
            for label, sequences in groups.items()
            if len(sequences) > 0}

        centers = {}
        variances = []
        for label, (center, variance) in bary_centers.items():
            variances += variance 
            centers[label] = center

        aligned = extract_alignment_points(groups, centers, instance_ids, variance_th=np.percentile(distances, 0.1), min_count=5)

        print("Training supervised model")
        all_vectors = []
        labels = []
        label_dict = {}
        label_id = 0
        for cluster, points in aligned.items():
            for point, ids in points.items(): 
                key = f"{cluster}::{point}"
                if key not in label_dict:
                    label_dict[key] = label_id
                    label_id += 1
                for i, j in ids:
                    all_vectors.append(raw_windows[i][j])
                    labels.append(label_dict[key])
        labels = np.array(labels)
        all_vectors = np.stack(all_vectors)
        n_labels = max(label_dict.values()) + 1
        print(f"Centers: {max(groups.keys())} Alignemnt: {n_labels}")        
        supervised = classifier(WINDOW_PARAM, encoder, n_labels)
        supervised.fit(all_vectors, labels, epochs=25)
        print("save models")
        encoder.save('{}/encoder_finetuning_epoch{}.h5'.format(out_folder, epoch))
        supervised.save('{}/supervised_alignment_points_epoch{}.h5'.format(out_folder, epoch))
        pkl.dump(label_dict, open('{}/labels{}.pkl'.format(out_folder, epoch), "wb"))

            
if __name__ == '__main__':
    print("=====================================")
    print("Simplified WDP DS Pipeline")
    print("by Daniel Kyu Hwa Kohlsdorf")
    if len(sys.argv) == 7 and sys.argv[1] == 'train':
        l1_labels = sys.argv[2]
        l1_wav    = sys.argv[3]
        l2_labels = sys.argv[4]
        l2_wav    = sys.argv[5]
        out       = sys.argv[6]
        train(l1_labels, l1_wav, l2_labels, l2_wav, out)
    elif len(sys.argv) == 6 and sys.argv[1] == 'aligned':
        encoder   = sys.argv[2]
        l2_labels = sys.argv[3]
        l2_wav    = sys.argv[4]
        out       = sys.argv[5]
        aligned(encoder, l2_labels, l2_wav, out)
    elif len(sys.argv) == 5 and sys.argv[1] == 'decode':
        classifier = sys.argv[2]
        audio      = sys.argv[3]
        label      = sys.argv[4]
        if audio.endswith('.wav'):
            output = audio.replace('.wav', '.csv')
            decode(classifier, audio, label, output)
        elif audio.endswith('/'):    
            for fp in os.listdir(audio):
                if fp.endswith('.wav'):
                    path = f"{audio}{fp}"
                    output = path.replace('.wav', '.csv')
                    print(f"Decoding: {path} {output}")
                    decode(classifier, path, label, output)
        else:
            print("Audio needs to be .wav or paths ending with /")            
    elif len(sys.argv) == 6 and sys.argv[1] == 'decode':
        classifier = sys.argv[2]
        audio      = sys.argv[3]
        label      = sys.argv[4]
        output     = sys.argv[5]
        decode(classifier, audio, label, output)
    elif len(sys.argv) == 5 and sys.argv[1] == 'extract':
        audio = sys.argv[2]    
        col = sys.argv[3]
        output = sys.argv[4]
        if audio.endswith('/'):    
            instance_id = 0
            for fp in os.listdir(audio):
                if fp.endswith('.wav'):
                    path = f"{audio}{fp}"
                    csv = path.replace('.wav', '.csv')
                    print(f"Extracting: {path} {output}")
                    instance_id = extract(path, csv, col, output, instance_id)
        else:
            print("Audio needs to be paths ending with /")            
    else:        
        print(sys.argv)
        print("""
            Usage:
                + train:      python pipeline.py train L1_CSV L1_AUDIO L2_CSV L2_AUDIO OUT_FOLDER
                + aligned:    python pipeline.py aligned ENCODER L2_CSV L2_AUDIO OUT_FOLDER
                + decode:     python pipeline.py decode CLASSIFIER (AUDIO|FOLDER) LABELS [OUTPUT]
                + extract:    python pipeline.py extract (AUDIO|FOLDER) COL OUTPUT
        """)
        print("\n=====================================")
