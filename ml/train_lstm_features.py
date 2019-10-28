import numpy as np 
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from audio import *
import os
import sys

def encoder(in_shape, target_dim):
    inp = Input(in_shape)
    x   = Conv2D(32, kernel_size=(3, 256), activation='relu', padding='same')(inp) 
    x   = Reshape((in_shape[0], in_shape[1] * 32))(x)
    x   = Bidirectional(LSTM(8, return_sequences=True))(x)
    x   = LSTM(target_dim)(x)    
    model = Model(inputs = [inp], outputs = [x])
    model.summary()
    return model

def predictor(target_dim, output_dim):
    inp = Input((target_dim))
    x   = Dense(output_dim, activation='linear')(inp)
    model = Model(inputs = [inp], outputs = [x])
    model.summary()
    return model

def auto_encoder(in_shape, latent_dim, output_dim):
    enc = encoder(in_shape, latent_dim)
    dec = predictor(latent_dim, output_dim)
    inp = Input(in_shape)
    x  = enc(inp) 
    x  = dec(x) 
    model = Model(inputs = [inp], outputs = [x])
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    return model, enc

def ae_from_file(paths, win, latent):    
    ae, enc = auto_encoder((win, 256, 1), latent, 256)
    w_before = enc.layers[1].get_weights()[0].flatten()
    data = [x for x in data_gen(paths, win, 'predict_next')]    
    x = np.stack([x for x, _ in data])
    y = np.stack([y for _, y in data])
    print(x.shape)
    print(y.shape)
    ae.fit(x = x, y = y, batch_size = 10, shuffle = True, epochs = 128)
    w_after = enc.layers[1].get_weights()[0].flatten()
    print("DELTA W:", np.sum(np.square(w_before - w_after)))
    return enc, ae

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python train_lstm_features.py WIN LATENT_DIM FOLDER1 ... FOLDERN")
    else:
        win = int(sys.argv[1])
        dim = int(sys.argv[2])
        folders = sys.argv[3:]
        encoder, ae = ae_from_file(folders, win, dim)
        encoder.save('encoder.h5') 
        ae.save('autoencoder.h5')