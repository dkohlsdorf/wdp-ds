import re
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

from collections import namedtuple

from lib_dolphin.audio import *
from lib_dolphin.interest_points import *

from scipy.io.wavfile import read, write    
from matplotlib.patches import Rectangle


def label(x, label_dict):
    n_labels = len(label_dict)
    n_labels = np.zeros(n_labels)
    for l in x:
        n_labels[l] += 1
    return label_dict[np.argmax(n_labels)]


def export_audio(c, labels, windows, label_dict, out, min_support = 25):
    windows_by_cluster = {}
    labels_by_cluster = {}
    for i, k in enumerate(c):
        win = windows[i]
        lab = labels[i]
        if k not in windows_by_cluster:
            windows_by_cluster[k] = []
            labels_by_cluster[k] = []
        windows_by_cluster[k].append(win)
        labels_by_cluster[k].append(lab)

    reverse = dict([(v, k) for k, v in label_dict.items()])
    
    whitelist = {}
    cur = 0
    for c, instances in windows_by_cluster.items():
        l = label(labels_by_cluster[c], reverse)
        audio = []
        if len(instances) > min_support:
            whitelist[c] = cur
            cur += 1
            for instance in instances:
                for x in instance:
                    audio.append(x)
                for i in range(0, 1000):
                    audio.append(0)
            audio = np.stack(audio)
            write('{}/{}_{}.wav'.format(out, l, whitelist[c]), 44100, audio.astype(np.int16)) 
    return whitelist
    

def plot_result_matrix(confusion, classes, predictions, title, cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion.shape[1]),
           yticks=np.arange(confusion.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=predictions, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.1f'
    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, format(confusion[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_tensorflow_hist(hist, output):
    if 'mse' in hist.history:
        plt.plot(hist.history['mse'],     label="mse")
        plt.plot(hist.history['val_mse'], label="val_mse")
        plt.title("MSE")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('mse')
    elif 'accuracy' in hist.history:
        plt.plot(hist.history['accuracy'],     label="accuracy")
        plt.plot(hist.history['val_accuracy'], label="val_accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
    plt.savefig(output)
    plt.clf()


def visualize_dataset(instances, output):
    plt.figure(figsize=(20, 20))
    for i in range(0, 100):
        xi = np.random.randint(0, len(instances))
        plt.subplot(10, 10, i + 1)
        plt.imshow(1.0 - instances[xi].T, cmap='gray')
    plt.savefig(output)
    plt.clf()

    
def reconstruct(ae, instances, output):    
    plt.figure(figsize=(5, 10))
    t = instances[0].shape[0]
    d = instances[0].shape[1]
    for i in range(25):
        idx = np.random.randint(len(instances))
        reconstruction = ae.predict(instances[idx].reshape((1, t, d, 1)))
        reconstruction = reconstruction.reshape((t, d))
        plt.subplot(5, 5, i + 1)
        plt.imshow(reconstruction.T)
    plt.savefig(output)
    plt.clf()   


def enc_filters(enc, n_filters, output):
    plt.figure(figsize=(10, 10))
    w = enc.weights[0].numpy()
    for i in range(w.shape[-1]):
        weight = w[:, :, 0, i]
        plt.subplot(n_filters / 8, 8, i + 1)
        plt.imshow(weight.T, cmap='gray')
    plt.savefig(output)
    plt.clf()


def decoded_plots(clustered, names, counts, path, ip_th, ip_r, show_ip=False):
    colors = []
    for line in open('lib_dolphin/color.txt'):
        cmp = line.split('\t')
        colors.append(cmp[1].strip())

    by_file = {}
    for c, examples in clustered.items():
        for file, start, stop, _ in examples:
            if file not in by_file:
                by_file[file] = []
            by_file[file].append([c, start, stop])

    for file, annotations in by_file.items():
        print(file)
        x  = raw(file)
        s  = spectrogram(x, lo=0, hi=256)

        plt.figure(figsize=(len(s) / 100, 25))
        plt.imshow(1.0 - s.T, cmap='gray')
        if show_ip:
            ip = [p for p in interest_points(s, ip_r, ip_th)]
            plt.scatter([t for t, _ in ip], [f for _, f in ip], color='red')
        last = 0
        for i, (c, start, stop) in enumerate(annotations):
            color = colors[c]    
            start_spec = start / 128 
            stop_spec  = stop / 128                
            if counts[c] > 1:
                c = names[c]
                plt.gca().add_patch(Rectangle((start_spec, 0), (stop_spec - start_spec), 256, color=color, edgecolor='r', alpha=0.5))
                plt.gca().annotate('{}'.format(c), xy=(start_spec + (stop_spec - start_spec) / 2, 25))
            else:
                plt.gca().annotate('===', xy=(start_spec, 25))
                plt.gca().add_patch(Rectangle((start_spec, 0), (stop_spec - start_spec), 256, edgecolor='r', fill = None))

        img = '{}/{}'.format(path, file.split('/')[-1].replace('.wav', '.png'))
        plt.savefig(img)
        plt.close()
        
        
def distance_plots(distance, path):
    plt.imshow(distance)
    plt.savefig('{}/needleman.png'.format(path))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(distance.flatten(), bins=100)
    plt.savefig('{}/histogram.png'.format(path))
    plt.close()

    
def sequence_cluster_export(clustered, names, counts, path, sep='_'):
    clusters = []
    files    = []
    starts   = []
    stops    = []
    strings  = []
    for c, regions in clustered.items():
        if counts[c] > 1:
            c = names[c]
            audio = []
            for file, start, stop, s in regions[0:25]:
                cmp = file.replace('.wav', '').split('/')[-1].split(sep)
                if len(cmp) > 0 and len(cmp[1]) > 0:
                    cmp[0] = re.sub("[^0-9]", "", cmp[0])
                    cmp[1] = re.sub("[^0-9]", "", cmp[1])
                    enc = int(cmp[0])
                    sec = int(cmp[1])

                    clusters.append(c)
                    files.append(enc)
                    starts.append(start / 44100 + sec)
                    stops.append(stop   / 44100 + sec)
                    strings.append(s)
                    x = raw(file)[start:stop]

                    for sample in x:
                        audio.append(sample)
                    for _ in range(0, 1000):
                        audio.append(0)
            audio = np.stack(audio)
            write('{}/cluster_{}.wav'.format(path, c), 44100, audio.astype(np.int16)) 
    df = pd.DataFrame({
        "file": files,
        "start": starts,
        "stop": stops,
        "cluster": clusters,
        "categories": strings
    })
    df.to_csv('{}/sequnces.csv'.format(path))
