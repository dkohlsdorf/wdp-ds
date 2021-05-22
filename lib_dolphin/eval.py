import re
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

from collections import namedtuple
from lib_dolphin.audio import *
from scipy.io.wavfile import read, write    
    

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

    
def enc_filters(enc, n_filters, output):
    plt.figure(figsize=(10, 10))
    w = enc.weights[0].numpy()
    for i in range(w.shape[-1]):
        weight = w[:, :, 0, i]
        plt.subplot(n_filters / 8, 8, i + 1)
        plt.imshow(weight.T, cmap='gray')
    plt.savefig(output)
    plt.clf()


        
        
    