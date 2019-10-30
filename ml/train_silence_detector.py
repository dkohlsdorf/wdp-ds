
import sys
import numpy as np
import matplotlib.pyplot as plt

from random import random

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.initializers import Constant
from audio import *

def detector(win, encoder):
    shape = encoder.layers[0].input_shape[0][1:]
    inp = Input(shape)
    x   = encoder(inp)
    x   = BatchNormalization()(x)    
    x   = Dense(64)(x)
    x   = Dense(32)(x)
    x   = Dropout(0.5)(x) 
    x   = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = [inp], outputs = [x])
    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

def label(x):
    if x.startswith('noise'):
        return 1.0
    else:
        return 0.0
    
def accept(y):
    if y == 1 and random() < 0.4 or y == 0:
        return True
    else:
        return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python train_silence_detector.py WIN FOLDER1 ... FOLDERN")
    else:
        win = int(sys.argv[1])
        folders = sys.argv[2:]
        
        encoder = load_model('models/lstm_v3/v3.2/encoder.h5')
        for layer in encoder.layers[:-1]:
            layer.trainable = False
        encoder.summary()
        noise_classifier = detector(win, encoder)
        sampled = [(x, y) for x, y in data_gen(folders, win, lambda x: x.startswith('noise')) if accept(y)]
        x = np.stack([x for x, _ in sampled])
        y = np.array([y for _, y in sampled])
        c = [0, 0]
        for i in y:
            c[int(i)] += 1
        print(c)
        noise_classifier.fit(x = x, y = y, shuffle=True, epochs = 64, batch_size=10)  
        noise_classifier.save('sil.h5')
