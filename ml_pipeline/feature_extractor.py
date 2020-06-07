# Deep Unsupervised Feature Learning
#  
# REFERENCES: 
# [KOH4] Daniel Kohlsdorf, Denise Herzing, Thad Starner: "An Auto Encoder For Audio Dolphin Communication", IJCNN, 2020

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import * 
from tensorflow.keras.losses import * 

KERNEL_SIZE = (8,8)
N_FILTERS   = 1024


def encoder(in_shape, latent_dim):
    """
    A LSTM stack on top of a convolutional layer pooled in time.

    See Figure [KOH4] Figure 2.

    :param in_shape: the input shape to the model
    :param latent_dim: embedding size of the model

    :returns: a keras model
    """
    dft_dim = in_shape[1]
    inp = Input(in_shape)
    loc = Conv2D(N_FILTERS, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(inp) # Shape (Time, DFT, Filters)
    loc = MaxPool2D(pool_size=(1, dft_dim))(loc) # Pool in time (Time, DFT, Filters) -> (Time, 1, Filters)
    loc = Reshape((in_shape[0], N_FILTERS))(loc) # Reshape for temporal model (Time, 1, Filters)  -> (Time, Filters)
    x   = BatchNormalization()(loc)
    x   = Bidirectional(LSTM(latent_dim, return_sequences=True))(x)
    x   = LSTM(latent_dim)(x)            
    return Model(inputs =[inp], outputs=[x])


def decoder(length, latent_dim, output_dim):
    """
    A LSTM stack followed by a de-convolution layer to reconstruct the input

    :param length: length of the sequence to reconstruct
    :param latent_dim: dimension of the latent space we reconstruct from
    :param output_dim: dimension of output

    See Figure [KOH4] Figure 2.

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


def auto_encoder(in_shape, latent_dim):
    """
    Auto encoder from encoder / decoder architecture on top of
    convolution / deconvolution layers

    See Figure [KOH4] Figure 2.

    :param in_shape: input shape (time, dimensions, 1)
    :param latent_dim: the length of the embedding vector

    :returns: a keras model for the auto encoder and a separate for the encoder
    """
    enc = encoder(in_shape, latent_dim)
    dec = decoder(in_shape[0], latent_dim, in_shape[1])
    inp = Input(in_shape)
    x   = enc(inp) 
    x   = dec(x) 
    model = Model(inputs = [inp], outputs = [x])
    model.compile(optimizer = RMSprop(), loss='mse')
    return model, enc

