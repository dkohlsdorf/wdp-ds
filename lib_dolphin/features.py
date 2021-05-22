# Deep Unsupervised Feature Learning
#  
# REFERENCES: 
# [KOH4] Daniel Kohlsdorf, Denise Herzing, Thad Starner: "An Auto Encoder For Audio Dolphin Communication", IJCNN, 2020

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import * 
from tensorflow.keras.losses import * 
from tensorflow.keras.regularizers import * 


def encoder(in_shape, latent_dim, conv_params):
    kernel_size = (conv_params[0], conv_params[1])
    n_filters = conv_params[2]
    dft_dim = in_shape[1]
    inp = Input(in_shape)
    loc = Conv2D(n_filters, strides = (1, 1), kernel_size=kernel_size, activation='relu', padding='same')(inp) 
    loc = MaxPool2D(pool_size=(1, dft_dim))(loc) 
    loc = Reshape((in_shape[0], n_filters))(loc) 
    x   = Bidirectional(LSTM(latent_dim, return_sequences=True))(loc)
    x   = LSTM(latent_dim)(x)            
    return Model(inputs =[inp], outputs=[x])


def classifier(in_shape, latent_dim, out_dim, conv_params):
    enc = encoder(in_shape, latent_dim, conv_params)
    inp = Input(in_shape)
    x   = enc(inp)
    x   = Dropout(0.5)(x) 
    x   = Dense(out_dim, activation='softmax')(x) 
    model = Model(inputs = [inp], outputs = [x])
    return model, enc