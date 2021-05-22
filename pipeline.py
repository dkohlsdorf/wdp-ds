import json
import numpy as np
import pickle as pkl
import sys
import os

from lib_dolphin.audio import *
from lib_dolphin.features import *
from lib_dolphin.eval import *

from collections import namedtuple

from scipy.io.wavfile import read, write

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
LATENT       = 128
BATCH        = 25
EPOCHS       = 25


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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    else:
        print("""
            Usage:
                + train:     python pipeline.py train LABEL_FILE AUDIO_FILE NOISE_FILE OUT_FOLDER
        """)
    print("\n=====================================")
