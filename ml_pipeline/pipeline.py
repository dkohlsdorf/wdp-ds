import sys

from tensorflow.keras.models import load_model

from audio import *
from feature_extractor import *
from classifier import *
from plots import *


def no_label(f,x):
    '''
    Return none for no label
    '''
    return None


def sil(f, x):
    ''' 
    For silence classification return 
    positive label if the file starts with noise
    '''
    if f.split('/')[-1].startswith('noise'):
        return 1.0
    else:
        return 0.0


def auto_encode(f, x):
    '''
    For auto encoding the label is the spectrogram itself
    '''
    return x


def train(folder, params, lable, model, batch_size=10, epochs=128):
    '''
    Train the model for some epochs with a specific batch size
    
    data: a data iterator
    model: a keras model
    batch_size: size of the mini batch
    epochs: number of runs over the complete dataset
    '''
    for epoch in range(epochs):
        batch = []
        for (x, y, _, _, _) in dataset(folder, params, lable, True):
            batch.append((x,y))
            total_loss = 0.0
            if len(batch) == batch_size:
                x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for x, _ in batch])
                y = np.stack([y.reshape(y.shape[0], y.shape[1], 1) for _, y in batch])
                total_loss += model.train_on_batch(x=x, y=y)
                batch = []                        
        print("EPOCH: {} LOSS: {}".format(epoch, total_loss))


def train_auto_encoder(version_tag, input_folder, output_folder, params, latent, batch, epochs):
    '''
    Train an auto encoder for feature embedding

    version_tag: basically the model name
    input_folder: the folder with the training data
    output_folder: the folder to save the model
    params: window parameters
    latent: dimension of the latent space
    batch: batch size
    epochs: number of training epochs
    '''
    print("Training Auto Encoder: {}".format(version_tag))
    ae, enc     = auto_encoder(
        (params.spec_win, params.n_fft_bins, 1), latent
    )
    enc.summary()
    w_before = enc.layers[1].get_weights()[0].flatten()
    train(input_folder, params, auto_encode, ae, batch, epochs)
    w_after = enc.layers[1].get_weights()[0].flatten()
    print("DELTA W:", np.sum(np.square(w_before - w_after)))
    enc.save('{}/encoder.h5'.format(output_folder))
    ae.save('{}/auto_encoder.h5'.format(output_folder))


def evaluate_encoder(version_tag, input_folder, output_folder, encoder_file, params, k):
    '''
    Evaluate an encoder for feature embedding

    version_tag: basically the model name
    input_folder: the folder with the data to embed
    output_folder: the folder to save the plots
    encoder_file: a saved encoder
    params: window parameters
    k: number of clusters
    '''
    print("Evaluate Encoder: {}".format(version_tag))
    enc = load_model(encoder_file)
    visualize_2dfilters(output_folder, enc, [1], n_rows = 8)    
    x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for (x,_,_,_,_) in dataset(
        folder, params, no_label, False
    )])
    visualize_embedding("{}/embeddings.png".format(output_folder), x, enc, k)


params      = WindowParams(128, 64, 512, 64, 25)
latent      = 128 
batch       = 10
epochs      = 2
folder      = 'data/classification_noise'
k           = 12
version     = "v1.1"
train_auto_encoder(version, folder, "./", params, latent, batch, epochs)
evaluate_encoder(version, folder, "./", "encoder.h5", params, k)
