import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import os

from collections import Counter
from collections import namedtuple
from lib_dolphin.audio import *
from lib_dolphin.htk_helpers import * 
from scipy.io.wavfile import read, write    

from matplotlib.colors import Normalize


COLORS = list(
    pd.read_csv('lib_dolphin/colors.txt', sep='\t', header=None)[1].apply(lambda x: x + "80")
)


BIAS   = 0.7
START  = 0.2
STOP   = 0.9
SCALER = 1.0


def plot_annotations(anno_files, labels, wav_folder, out_folder, win, th, noise_th = 0.99, plot_noise = False, do_compress=False):
    n = -1
    filtered = {}
    for file, annotations in anno_files.items():
        n += 1
        if len(annotations) > 1 and do_compress:
            annotations = compress(annotations)
        if len(annotations) > 1:
            filtered[file] = []
            path = "{}/{}.wav".format(wav_folder, file)
            x = raw(path)
            s = spectrogram(x, lo = 0, hi = 256)

            max_annotations = max([stop for _, stop, _, _ in annotations])
            lab_df = pd.read_csv(labels[file])
            print(file, n, len(s), len(lab_df), max_annotations)
            if len(s) < 10000:
                fig, ax = plt.subplots()
                fig.set_size_inches(len(s) / 100, len(s[0]) / 100)
                ax.imshow(BIAS - s.T * SCALER, norm=Normalize(START, STOP), cmap='gray')
                for start, stop, i, ll in annotations:
                    if ll >= th:
                        label_regions = lab_df['labels'][start:stop]
                        counter = Counter(label_regions)
                        n_noise = counter['NOISE'] 
                        n_not_noise = len(label_regions) - n_noise
                        ratio = n_noise / (n_not_noise + n_noise)
                        
                        is_noise = n_not_noise == 0 or ratio > noise_th
                        if not is_noise or plot_noise:
                            filtered[file].append((start, stop, i, ll))
                            a = start * win
                            e = stop  * win
                            plt.text(a + (e - a) // 2 , 30, i, size=20)
                            if is_noise:
                                plt.text(a, 30, "N", size=20)                        
                            rect = patches.Rectangle((a, 0), e - a, 256, linewidth=1, edgecolor='r', facecolor=COLORS[i])
                            ax.add_patch(rect)
                plt.savefig("{}/{}.png".format(out_folder, file))
                plt.close()
            else:
                print("\t skip")
    return filtered


def plot_result_matrix(confusion, classes, predictions, title, cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    im = ax.imshow(confusion, interpolation='nearest', cmap=cmap)
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

    
def enc_filters(enc, n_filters, output):
    plt.figure(figsize=(10, 10))
    w = enc.weights[0].numpy()
    for i in range(w.shape[-1]):
        weight = w[:, :, 0, i]
        plt.subplot(n_filters / 8, 8, i + 1)
        plt.imshow(weight.T, cmap='gray')
    plt.savefig(output)
    plt.clf()
