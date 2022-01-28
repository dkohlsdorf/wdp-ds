import pickle as pkl
import sys
import time


from collections import namedtuple
from lib_dolphin.audio import *
from tensorflow.keras.models import load_model


SPLIT_SEC    = 60
SPLIT_RATE   = 44100
SPLIT_SKIP   = 0.5

FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 230
FFT_LO       = 100

D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)


NEURAL_NOISE_DAMPENING = 0.5
NEURAL_SMOOTH_WIN      = 64
NEURAL_SIZE_TH         = 32


def split(audio_file):
    window_size = SPLIT_SEC * SPLIT_RATE
    skip        = int(window_size * SPLIT_SKIP)
    x           = raw(audio_file)
    n           = len(x)

    regions = []
    for i in range(window_size, n, skip):
        regions.append(x[i-window_size:i])
    return regions


def spec(x):
    return spectrogram(x, FFT_LO, FFT_HI, FFT_WIN, FFT_STEP)


def decode(x, decoder, label_mapping):
    a = x.reshape((1, len(x), D, 1))
    p = decoder.predict(a).reshape((a.shape[1], label_mapping.n + 1)) 
    if len(p) > NEURAL_SMOOTH_WIN:
        for i in range(0, len(p[0])):
            p[:, i] = np.convolve(p[:, i], np.ones(NEURAL_SMOOTH_WIN) / NEURAL_SMOOTH_WIN, mode='same')
    p[:, 0] *= NEURAL_NOISE_DAMPENING
    local_c = p.argmax(axis=1)
    return local_c


def sequence_building(decoded):
    pass


if __name__ == '__main__':
    print("Decoder")    
    decoder  = load_model('../data/decoder_nn.h5')
    lab      = pkl.load(open("../data/labels.pkl", "rb"))
    reverse  = {v:k for k, v in lab.items()}
    label_mapping = pkl.load(open('../data/label_mapping.pkl', 'rb'))

    start = time.time()
    filename = "../data/dolphin.wav"
    x = split(filename)
    for i in range(len(x)):
        spec = spec(x[i])
        dec  = decode(s, decoder, label_mapping)
        seq  = sequence_building(dec)
        
        if i % 10 == 0 and i > 0:
            stop = time.time()
            secs = stop - start
            print("Execute 10 minutes {} [seconds]".format(int(secs)))
            start = time.time()
    
