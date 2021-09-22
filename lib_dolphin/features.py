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
    shape = (None, dft_dim, 1)
    inp = Input(shape)
    loc = Conv2D(n_filters, strides = (1, 1), kernel_size=kernel_size, activation='relu', padding='same')(inp) 
    loc = MaxPool2D(pool_size=(1, dft_dim))(loc) 
    loc = Reshape((-1, n_filters))(loc) 
    x   = Bidirectional(LSTM(latent_dim, return_sequences=True))(loc)
    return Model(inputs =[inp], outputs=[x])


def seq2seq_classifier(in_shape, encoder, n_latent, n_classes):    
    dft_dim = in_shape[1]
    shape = (None, dft_dim, 1)
    inp   = Input(shape)
    x     = encoder(inp) 
    x     = LSTM(n_latent, return_sequences=True)(x)            
    x     = TimeDistributed(Dropout(0.5))(x) 
    x     = TimeDistributed(Dense(n_classes, activation='softmax'))(x)
    model = Model(inputs = [inp], outputs = [x])
    return model    
    

def window_encoder(in_shape, encoder, latent_dim):
    inp   = Input(in_shape)
    x     = encoder(inp) 
    x     = LSTM(latent_dim)(x)            
    model = Model(inputs = [inp], outputs = [x])
    return model


def decoder(length, latent_dim, output_dim, conv_params):
    kernel_size = (conv_params[0], conv_params[1])
    n_filters = conv_params[2]
    inp = Input((latent_dim))
    x   = Reshape((1, latent_dim))(inp)
    x   = ZeroPadding1D((0, length - 1))(x)
    x   = LSTM(latent_dim, return_sequences=True)(x)    
    x   = Bidirectional(LSTM(output_dim // 2, return_sequences=True))(x)
    x   = Reshape((length, output_dim, 1))(x)
    x   = Conv2DTranspose(n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x) 
    x   = Conv2DTranspose(1, kernel_size=(1, 1), activation='linear', padding='same')(x) 
    return Model(inputs = [inp], outputs = [x])


def auto_encoder(in_shape, encoder, latent_dim, conv_params):
    dec = decoder(in_shape[0], latent_dim, in_shape[1], conv_params)
    inp = Input(in_shape)
    x   = encoder(inp) 
    x   = dec(x) 
    model = Model(inputs = [inp], outputs = [x])
    model.compile(optimizer = RMSprop(), loss='mse')
    return model


def classifier(in_shape, enc, latent_dim, out_dim, conv_params):
    inp = Input(in_shape)
    x   = enc(inp)
    x   = Dropout(0.5)(x) 
    x   = Dense(out_dim, activation='softmax')(x) 
    model = Model(inputs = [inp], outputs = [x])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class TripletLoss(Loss):

    def __init__(self, margin, latent):
        super().__init__()
        self.margin = margin
        self.latent = latent

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        anchor = y_pred[:, 0:self.latent]
        pos    = y_pred[:, self.latent:2 * self.latent]
        neg    = y_pred[:, 2 * self.latent:3 * self.latent]
        pos_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)            
        neg_dist   = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)    
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.margin)    
        loss       = tf.reduce_sum(tf.maximum(basic_loss, 0.0))               
        return loss


def triplet_model(in_shape, encoder, latent, margin=1.0):
    i = Input(in_shape)
    e = encoder(i)
    e = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(e)
    encoder = tf.keras.models.Model(inputs=i, outputs=e)  

    anchor = Input(in_shape)
    pos    = Input(in_shape)
    neg    = Input(in_shape)
    z_a    = encoder(anchor)
    z_p    = encoder(pos)
    z_n    = encoder(neg)
    conc   = Concatenate()([z_a, z_p, z_n])

    model   = tf.keras.models.Model(inputs=[anchor, pos, neg], outputs=conc)  
    triplet_loss = TripletLoss(margin, latent)
    model.compile(optimizer = 'adam', loss = triplet_loss)
    return model
    
