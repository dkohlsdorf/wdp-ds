import sys
import yaml
import glob
import numpy as np
import subprocess  
import os
import datetime
import tensorflow as tf
import pickle as pkl
import matplotlib
matplotlib.use('Agg')
import multiprocessing as mp
import re
import logging
logging.basicConfig()
log = logging.getLogger('main')
log.setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model
from feature_extractor import *
from sklearn.cluster import *

from plots import *
from audio_collection import *
from audio import * 


def train(folder, output_folder, params, enc, ae, batch_size=10, epochs=128, keep=lambda x: True):
    """
    Train the model for some epochs with a specific batch size

    :param data: a data iterator
    :param model: a keras model
    :param batch_size: size of the mini batch
    :param epochs: number of runs over the complete dataset
    :param keep: function from label to keep or not
    """
    n_processed = 0
    training_log = open('{}/loss.csv'.format(output_folder), 'w') 
    for epoch in range(epochs):
        batch = []
        epoch_loss = 0.0
        for (x, _, _, _) in dataset(folder, params, True):
            batch.append(x)
            total_loss = 0.0
            if len(batch) == batch_size:
                x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for x in batch])
                loss = ae.train_on_batch(x=x, y=x)
                total_loss += loss
                batch = []
                if n_processed % 10 == 0:
                    log.info("#: {} EPOCH: {} LOSS: {}".format(n_processed, epoch, total_loss))
                    total_loss = 0.0
                n_processed += 1
                epoch_loss += loss
        training_log.write('{},{},{}\n'.format(epoch, n_processed, epoch_loss))
        training_log.flush()
    training_log.close()
    

def train_auto_encoder(version_tag, input_folder, output_folder, params, latent, batch, epochs, conv_param):
    """
    Train an auto encoder for feature embedding

    :param version_tag: basically the model name
    :param input_folder: the folder with the training data
    :param output_folder: the folder to save the model
    :param noise_folder: folder some files with only noise for data augmentation
    :param params: window parameters
    :param latent: dimension of the latent space
    :param batch: batch size
    :param epochs: number of training epochs
    :param conv_params: (conv_w, conv_h, filters)
    """
    log.info("Training Auto Encoder: {}".format(version_tag))
    ae, enc = auto_encoder(
        (params.spec_win, params.n_fft_bins, 1), latent, conv_param
    )
    enc.summary()
    if os.path.exists('{}/encoder.h5'.format(output_folder)) and os.path.exists('{}/auto_encoder.h5'.format(output_folder)):
        log.info("\tloading previous weights")
        _enc = load_model('{}/encoder.h5'.format(output_folder))
        _enc.summary()
        _ae  = load_model('{}/auto_encoder.h5'.format(output_folder))
        enc.set_weights(_enc.get_weights())
        ae.set_weights(_ae.get_weights())
    w_before = enc.layers[1].get_weights()[0].flatten()
    train(input_folder, output_folder, params, enc, ae, batch, epochs)
    w_after = enc.layers[1].get_weights()[0].flatten()
    log.info("DELTA W:", np.sum(np.square(w_before - w_after)))
    enc.save('{}/encoder_reconstruction.h5'.format(output_folder), include_optimizer=False)
    enc.save('{}/encoder.h5'.format(output_folder), include_optimizer=False)
    ae.save('{}/auto_encoder.h5'.format(output_folder), include_optimizer=False)

    
def test_reconstruction(folder, out, params):
    """
    Reconstruct 100 examples using the auto encoder
    """
    log.info("Testing Reconstruction")
    ae = load_model('{}/auto_encoder.h5'.format(out))
    gen = dataset(folder, params, True)
    i = 0
    plt.figure(figsize=(40, 40))
    for (x, f, _, _) in gen:
        name = f.split('/')[-1]

        plt.subplot(10, 10, i + 1)
        plt.axis('off')
        plt.imshow(1.0 - ae.predict(x.reshape(1, params.spec_win, params.n_fft_bins, 1))[0, :, :, 0].T, cmap='gray')
        i += 1
        if i % 10 == 0:        
            log.info(i)
        if i == 100:
            break
    plt.savefig('{}/reconstructions.png'.format(out))
    plt.close()


def write_audio(out, prefix, cluster_id, grouped_by_cluster):
    """
    Write clusters as audio
    
    :param cluster_id: id to write
    :param instances_clusters: number of instances per cluster
    :param grouped_by_cluster: dict[clusters][filename][start, stop]
    :param returns: true if we stopped writing early
    """
    x = [a for a in grouped_by_cluster[cluster_id].items()]

    audio_bank = AudioSnippetCollection("{}/{}_seq_cluster_{}.wav".format(out, prefix, cluster_id))
    n_written = 0    
    for f, snippets in x:
        for audio_snippet in audio_regions(f, snippets):                  
            audio_bank.write(audio_snippet)
            n_written += 1
    audio_bank.close()
    log.info("Done: {}".format(cluster_id))


def evaluate_encoder(version_tag, input_folder, output_folder, encoder_file, params, min_support=5):
    """
    Evaluate an encoder for feature embedding
    :param version_tag: basically the model name
    :param input_folder: the folder with the data to embed
    :param output_folder: the folder to save the plots
    :param encoder_file: a saved encoder
    :param params: window parameters
    :param min_support: minimum support for a pattern
    """
    log.info("Evaluate Encoder: {}".format(version_tag))
    enc = load_model(encoder_file)
    visualize_2dfilters(output_folder, enc, [1], n_rows = 8)    
    
    data   = [tuples for tuples in dataset(input_folder, params, False)]
    log.info("#Input: {}".format(len(data)))

    stds   = [np.std(x) for (x,_,_,_) in data]
    std_th = np.percentile(stds, 75)
    data   = [d for d, std in zip(data, stds) if std > std_th]
    log.info("#Input after filtering {}: {}".format(std_th, len(data)))

    x      = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for (x,_,_,_) in data])
    h     = enc.predict(x)
    mu_h  = np.mean(h, axis=1)
    std_h = np.std(h, axis=1) 
    h     = ((h.T - mu_h) / std_h).T

    distances = []
    for _ in range(0, 5000):
        i = np.random.randint(0, len(h))
        j = np.random.randint(0, len(h))
        d = np.sqrt(np.sum(np.square(h[i] - h[j])))
        distances.append(d)
    th = np.percentile(distances, 50)
    log.info("Clustering with: {} threshold: {}".format(h.shape, th))
    clustering = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=th)
    c = clustering.fit_predict(h)

    visualize_embedding("{}/embeddings.png".format(output_folder), h, x, c)    

    grouped_by_filename = {}
    grouped_by_cluster  = {}
    i = 0
    k = 0
    for (_, f, start, stop), c in zip(data, c):
        if c not in grouped_by_cluster:
            grouped_by_cluster[c] = {}
        if f not in grouped_by_cluster[c]:
            grouped_by_cluster[c][f] = []
        grouped_by_cluster[c][f].append((start, stop))
        if f not in grouped_by_filename:
            grouped_by_filename[f] = []
        grouped_by_filename[f].append((start, stop, c, h[i]))
        if c > k:
            k = c
        i += 1
    k = k + 1
    
    log.info('n clusters: {}'.format(k))
    instances_clusters = np.zeros(k, dtype=np.int32)
    for c, collection in grouped_by_cluster.items():
        for f, regions in collection.items():
            for r in regions:
                instances_clusters[c] += 1

    log.info('Done Clustering')
    with mp.get_context("spawn").Pool(processes=5) as pool: 
        pool.starmap(write_audio, ((output_folder, "clusters", cluster_id, grouped_by_cluster) for cluster_id in range(0, k) if instances_clusters[cluster_id] >= min_support))
    log.info('Done Writing')

    for f, regions in grouped_by_filename.items():
        filename = f.split(".")[0].split("/")[-1]
        log_path = "{}/{}_clustering_log_{}.csv".format(output_folder, "clusters", filename)
        log.info("writing: {}".format(log_path))
        with open(log_path, "w") as fp:
            fp.write("start\tstop\tfile\tcluster\tembedding\n")
            for start, stop, c, h in regions:
                if instances_clusters[c] >= min_support:
                    vector = ",".join([str(x) for x in h.flatten()])
                    fp.write("{}\t{}\t{}\t{}\t{}\n".format(start, stop, f, c, vector))
    log.info('Done Logs')
    

def header():
    return """
    =================================================================
    Dolphin Machine Learning Pipeline
                
    usage: python ml_pipeline/pipeline.py config/default_config.yaml

    by Daniel Kyu Hwa Kohlsdorf
    =================================================================
    """


if __name__== "__main__":
    log.info(header())
    if len(sys.argv) == 2: 
        c = yaml.load(open(sys.argv[1]))
        log.info("Parameters: {}".format(c))
        version      = c['version']

        # spectrogram parameters
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        
        # neural parameters
        latent          = c['latent']
        batch           = c['batch']
        epochs          = c['epochs']
        conv_param      = (c['conv_w'],  c['conv_h'],  c['conv_filters'])
        
        # datasets
        unsupervised = c['unsupervised']
        output       = c['output']

        # clutering paams
        log.info("Mixed Training Epoch AE")
        train_auto_encoder(version, unsupervised, output, params, latent, batch, epochs, conv_param)
        evaluate_encoder(version, unsupervised, output, "{}/encoder.h5".format(output), params)       
        test_reconstruction(unsupervised, output, params)
        
