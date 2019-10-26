
import sys
import numpy as np
import matplotlib.pyplot as plt

from random import random
from convnet import data_gen
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def detector(win, encoder):
    inp = Input((win, 256, 1))
    x   = encoder(inp)
    x   = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = [inp], outputs = [x])
    model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python convnet.py WIN FOLDER1 ... FOLDERN")
    else:
        win = int(sys.argv[1])
        folders = sys.argv[2:]
        
        encoder = load_model('encoder.h5')
        for layer in encoder.layers:
            layer.trainable = False
        encoder.summary()
        noise_classifier = detector(win, encoder)
        x = np.stack([x for x, _ in data_gen(folders, win, lambda x: x.startswith('noise'))])
        y = [y for _, y in data_gen(folders, win, lambda x: x.startswith('noise'))]
        noise_classifier.fit(x = x, y = y, shuffle=True, epochs = 64, batch_size=10)  
        noise_classifier.save('sil.h5')
