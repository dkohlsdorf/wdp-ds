import numpy as np 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from audio import *
import os
import sys

class VAE:

    def encoder(self, in_shape, latent_dim):
        self.enc_inp = Input(in_shape)
        loc = Conv2D(256, kernel_size=(8, 8), activation='relu', padding='same')(self.enc_inp) 
        loc = MaxPool2D(pool_size=(1, 256))(loc)
        loc = Reshape((in_shape[0], 256))(loc)
        x   = BatchNormalization()(loc)
        x   = Bidirectional(LSTM(128, return_sequences=True))(x)
        x   = LSTM(32)(x)            
        self.mu      = Dense(latent_dim, activation = 'linear')(x)
        self.log_var = Dense(latent_dim, activation = 'linear')(x)        

        def sampling (args) : 
            mu , log_var = args 
            epsilon = K.random_normal(shape =K.shape(mu), mean = 0. , stddev = 1.)
            return mu + K.exp (log_var / 2) * epsilon

        self.encoder_output = Lambda(sampling, name ='encoder_output' )([self.mu, self.log_var]) 
        self.encoder        = Model(inputs =[self.enc_inp], outputs=[self.encoder_output])
        self.encoder.summary()

    def predictor(self, win, target_dim, output_dim):
        self.dec_inp = Input((target_dim))
        x = Reshape((1, target_dim))(self.dec_inp)
        x = ZeroPadding1D((0, win - 1))(x)
        x = LSTM(256, return_sequences=True)(x)    
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Reshape((win, 256, 1))(x)
        x = Conv2DTranspose(256, kernel_size=(8, 8), activation='relu', padding='same')(x) 
        self.dec_output = Conv2DTranspose(1,   kernel_size=(1, 1), activation='relu', padding='same')(x)     
        self.decoder = Model(inputs = [self.dec_inp], outputs = [self.dec_output])
        self.decoder.summary()

    def auto_encoder(self, in_shape, latent_dim, output_dim, win):
        self.encoder(in_shape, latent_dim)
        self.predictor(win, latent_dim, output_dim)

        inp = self.enc_inp
        out  = self.decoder(self.encoder_output) 
        self.model = Model(inputs = [inp], outputs = [out])
        self.model.summary()

        def vae_r_loss(y_true, y_pred ): 
            r_loss = K.mean(K.square(y_true - y_pred), axis = [ 1 , 2 , 3 ]) 
            return 2 * r_loss 

        def vae_kl_loss(y_true, y_pred): 
            kl_loss = -0.5 * K.sum( 1 + self.log_var -  K.square(self.mu) - K.exp(self.log_var), axis = 1 ) 
            return kl_loss
            
        def vae_loss(y_true, y_pred): 
            r_loss = vae_r_loss(y_true, y_pred) 
            kl_loss = vae_kl_loss(y_true, y_pred) 
            return r_loss + kl_loss 
        
        optimizer = Adam(lr = 0.01) 
        self.model.compile(optimizer=optimizer, 
                        loss=vae_loss, 
                        metrics = [ vae_r_loss , vae_kl_loss ],
                        experimental_run_tf_function=False)

def ae_from_file(paths, win, latent):    
    vae = VAE()
    vae.auto_encoder((win, 256, 1), latent, 256 * win, win)
    ae  = vae.model
    enc = vae.encoder 
    w_before = enc.layers[1].get_weights()[0].flatten()
    x = np.stack([x for x in data_gen(paths, win)])
    print(x.shape)
    ae.fit(x = x, y = x, batch_size = 10, shuffle = True, epochs = 32)
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