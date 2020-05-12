from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import * 
from tensorflow.keras.losses import * 
from tensorflow.keras import backend as K

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

KERNEL_SIZE = (8,8)
N_FILTERS   = 1024


def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon


class VAE:

    def encoder(self, in_shape, latent_dim):
        """
        A LSTM stack on top of a convolutional layer pooled in time.

        :param in_shape: the input shape to the model
        :param latent_dim: embedding size of the model

        :returns: a keras model
        """
        dft_dim = in_shape[1]
        inp = Input(in_shape)
        loc = Conv2D(N_FILTERS, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(inp) 
        loc = MaxPool2D(pool_size=(1, dft_dim))(loc)
        loc = Reshape((in_shape[0], N_FILTERS))(loc)
        x   = BatchNormalization()(loc)
        x   = Bidirectional(LSTM(latent_dim, return_sequences=True))(x)
        x   = LSTM(latent_dim)(x)       
        self.z_mean    = Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z              = Lambda(sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        return Model(inputs =[inp], outputs=[z])


    def decoder(self, length, latent_dim, output_dim):
        """
        A LSTM stack followed by a de-convolution layer to reconstruct the input

        :param length: length of the sequence to reconstruct
        :param latent_dim: dimension of the latent space we reconstruct from
        :param output_dim: dimension of output

        :returns: a keras model
        """
        inp = Input((latent_dim))
        x   = Reshape((1, latent_dim))(inp)
        x   = ZeroPadding1D((0, length - 1))(x)
        x   = LSTM(latent_dim, return_sequences=True)(x)    
        x   = Bidirectional(LSTM(output_dim // 2, return_sequences=True))(x)
        x   = Reshape((length, output_dim, 1))(x)
        x   = Conv2DTranspose(N_FILTERS, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x) 
        x   = Conv2DTranspose(1, kernel_size=(1, 1), activation='linear', padding='same')(x) 
        return Model(inputs = [inp], outputs = [x])

    def vae_r_loss(self, y_true , y_pred): 
        r_loss = K.mean(K.square(y_true - y_pred), axis = [ 1 , 2 , 3 ]) 
        return 10000.0 * r_loss 
    
    def vae_kl_loss(self, y_true, y_pred): 
        kl_loss = 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) -  K.exp(self.z_log_var), axis = 1)
        return kl_loss 

    def vae_loss(self, y_true, y_pred): 
        r_loss  = self.vae_r_loss(y_true, y_pred)
        kl_loss = self.vae_kl_loss(y_true, y_pred ) 
        return r_loss + kl_loss 

    def auto_encoder(self, in_shape, latent_dim):
        """
        Auto encoder from encoder / decoder architecture on top of
        convolution / deconvolution layers

        :param in_shape: input shape (time, dimensions, 1)
        :param latent_dim: the length of the embedding vector

        :returns: a keras model for the auto encoder and a separate for the encoder
        """
        enc = self.encoder(in_shape, latent_dim)
        dec = self.decoder(in_shape[0], latent_dim, in_shape[1])
        inp = Input(in_shape)
        e         = enc(inp)
        x         = dec(e) 
        model = Model(inputs = [inp], outputs = [x])
        model.compile(optimizer = Adam(), loss=self.vae_loss, metrics=[self.vae_kl_loss, self.vae_r_loss])
        model.summary()
        return model, enc

