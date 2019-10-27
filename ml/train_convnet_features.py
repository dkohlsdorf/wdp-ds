import numpy as np 
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from audio import *
import os
import sys

K_SZE   = (8, 8)
def conv(x, filters):
    x = Conv2D(filters, strides = (2, 2), kernel_size = K_SZE, activation='relu', padding='same')(x)
    return LeakyReLU()(x)

def unconv(x, filters, stride = (2,2), linear = False):
    x = Conv2DTranspose(filters, strides = stride, kernel_size = K_SZE, activation='relu', padding='same')(x)
    if linear:
        return x
    return LeakyReLU()(x)

def dense(x, dim):
    x = Dense(dim)(x)
    x = BatchNormalization()(x)
    return LeakyReLU()(x)

def flat(x):
    return Flatten()(x)

def reshape(x, shape):
    return Reshape(shape)(x)

def encoder(in_shape, target_dim):
    inp = Input(in_shape)
    x = conv(inp, 128)
    x = conv(x,   64)
    x = conv(x,   32) 
    before = conv(x, 16) 
    x = flat(before)
    x = dense(x, target_dim)
    model = Model(inputs = [inp], outputs = [x])
    model.summary()
    return model, before.shape

def decoder(dim, before_flat):
    N = 1.0
    shape = []
    for d in before_flat:
        if d is not None:
            N *= d
            shape.append(d)
    inp = Input((dim))
    x   = dense(inp, N)
    x   = reshape(x, shape)
    x = unconv(x, 16) 
    x = unconv(x, 32)
    x = unconv(x, 64)
    x = unconv(x, 128) 
    x = unconv(x, 1, stride = (1,1), linear=True) 
    model = Model(inputs = [inp], outputs = [x])
    model.summary()
    return model

def auto_encoder(in_shape, latent_dim):
    enc, shape = encoder(in_shape, latent_dim)
    dec = decoder(latent_dim, shape)
    inp = Input(in_shape)
    x  = enc(inp) 
    x  = dec(x) 
    model = Model(inputs = [inp], outputs = [x])
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    return model, enc

def data_gen(paths, win, mk_lable = None):
    frame = 0
    for path in paths:
        for file in os.listdir(path):
            if file.endswith('.wav'):
                lable = None
                if mk_lable is not None:
                    lable = mk_lable(file)
                print('process file {} {}'.format(file, frame))
                fp = "{}{}".format(path, file)
                spec = spectrogram_from_file(fp) 
                (t, d) = spec.shape
                for i in range(win, t, win // 2):
                    frame += 1
                    x = np.reshape(spec[i - win:i], (win,d,1))
                    mu  = np.mean(x)
                    std = np.std(x) + 1.0
                    if lable is None:
                        yield (x - mu) / std
                    else:
                        yield ((x - mu) / std, lable)
            
def ae_from_file(paths, win, latent):    
    ae, enc = auto_encoder((win, 256, 1), latent)
    w_before = enc.layers[1].get_weights()[0].flatten()
    x = np.stack([x for x in data_gen(paths, win)])
    ae.fit(x = x, y = x, batch_size = 10, shuffle = True, epochs = 128)
    w_after = enc.layers[1].get_weights()[0].flatten()
    print("DELTA W:", np.sum(np.square(w_before - w_after)))
    return enc, ae

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python convnet.py WIN LATENT_DIM FOLDER1 ... FOLDERN")
    else:
        win = int(sys.argv[1])
        dim = int(sys.argv[2])
        folders = sys.argv[3:]
        encoder, ae = ae_from_file(folders, win, dim)
        encoder.save('encoder.h5') 
        ae.save('autoencoder.h5')