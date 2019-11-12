import numpy as np 
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from audio import *
import os
import sys

def encoder(in_shape, latent_dim):
        inp = Input(in_shape)
        loc = Conv2D(128, kernel_size=(8, 8), activation='relu', padding='same')(inp) 
        loc = MaxPool2D(pool_size=(1, 256))(loc)
        loc = Reshape((in_shape[0], 128))(loc)
        x   = BatchNormalization()(loc)
        x   = Bidirectional(LSTM(128, return_sequences=True))(x)
        x   = LSTM(latent_dim)(x)            
        encoder    = Model(inputs =[inp], outputs=[x])
        encoder.summary()
        return encoder

def predictor(win, target_dim, output_dim):
    inp = Input((target_dim))
    x = Reshape((1, target_dim))(inp)
    x = ZeroPadding1D((0, win - 1))(x)
    x = LSTM(128, return_sequences=True)(x)    
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Reshape((win, 256, 1))(x)
    x = Conv2DTranspose(128, kernel_size=(8, 8), activation='relu', padding='same')(x) 
    x = Conv2DTranspose(1,   kernel_size=(1, 1), activation='linear', padding='same')(x) 
    model = Model(inputs = [inp], outputs = [x])
    model.summary()
    return model

def auto_encoder(in_shape, latent_dim, output_dim, win):
    enc = encoder(in_shape, latent_dim)
    dec = predictor(win, latent_dim, output_dim)
    inp = Input(in_shape)
    x  = enc(inp) 
    x  = dec(x) 
    model = Model(inputs = [inp], outputs = [x])
    model.summary()

    model.compile(optimizer='adam', loss='mse')
    return model, enc

def ae_from_file(paths, win, latent):    
    ae, enc  = auto_encoder((win, 256, 1), latent, 256 * win, win)
    w_before = enc.layers[1].get_weights()[0].flatten()
    x = np.stack([x for x in data_gen(paths, win)])
    ae.fit(x = x, y = x, batch_size = 100, shuffle = True, epochs = 128)
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
