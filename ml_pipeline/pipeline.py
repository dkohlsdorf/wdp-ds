import sys
import yaml

import numpy as np
import subprocess  
import os
import datetime
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import multiprocessing as mp

import logging
logging.basicConfig()
log = logging.getLogger('main')
log.setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model
from feature_extractor import *
from classifier import *
from plots import *
from sequence_embedder import *
from audio_collection import *
from audio import * 
from structured import *


def no_label(f,x):
    """
    Return none for no label

    :returns: None
    """
    return None


def lable(f, x):
    """
    Return label based on filename

    :returns: number for each filename 
    """
    if f.split('/')[-1][0] == 'n':
        return 0 # noise
    elif f.split('/')[-1][0] == 'e':
        return 1 # echo
    elif f.split('/')[-1][0] == 'b':
        return 2 # burst
    elif f.split('/')[-1][0] == 'w':
        return 3 # whistle


def sil(f, x):
    """
    For silence classification return
    positive label if the file starts with noise

    :returns: 1 if name starts with noise 0 otherwise
    """
    if f.split('/')[-1].startswith('noise'):
        return 1.0
    else:
        return 0.0


def auto_encode(f, x):
    """
    For auto encoding the label is the spectrogram itself

    :returns: spectrogram itself
    """
    return x


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
    history = []
    training_log = open('{}/loss.csv'.format(output_folder), 'w') 
    for epoch in range(epochs):
        batch = []
        epoch_loss = 0.0
        for (x, y, _, _, _) in dataset(folder, params, auto_encode, True):
            if keep(y):
                batch.append((x,y))
                total_loss = 0.0
                if len(batch) == batch_size:
                    x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for x, _ in batch])
                    y = np.stack([y.reshape(y.shape[0], y.shape[1], 1) for _, y in batch])
                    loss = ae.train_on_batch(x=x, y=y)
                    history.append(loss)
                    total_loss += loss
                    batch = []
                    if n_processed % 10 == 0:
                        log.info("#: {} EPOCH: {} LOSS: {}".format(n_processed, epoch, total_loss))
                        total_loss = 0.0
                    n_processed += 1
                    epoch_loss += loss
        training_log.write('{},{},{}\n'.format(epoch, n_processed, epoch_loss))
        training_log.flush()
        plt.plot(history)
        plt.savefig('{}/history_{}.png'.format(output_folder, epoch))
        enc.save('{}/encoder_{}.h5'.format(output_folder, epoch))
        ae.save('{}/auto_encoder_{}.h5'.format(output_folder, epoch))
    training_log.close()
    
    
def train_type(version_tag, input_folder, output_folder, params, encoder_file, batch, epoch, latent, freeze, transfer=True, subsample=0.5):
    """
    Train a multiclass type classifier
    :param version_tag: basically the model name
    :param input_folder: the folder with the training data
    :param output_folder: the folder to save the model
    :param params: window parameters
    :param encoder_file: a saved encoder
    :param batch: batch size
    :param epochs: number of training epochs
    """
    if transfer:
        log.info(encoder_file)
        enc = load_model(encoder_file)
    else:
        _, enc = auto_encoder(
            (params.spec_win, params.n_fft_bins, 1), latent
        )
    cls_type = classifier(enc, 4, freeze)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for (x, y, _, _, _) in dataset(input_folder, params, lable, True):
        if np.random.uniform() < subsample and x is not None and y is not None:
            if np.random.uniform() > 0.6:                
                x_test.append(x)
                y_test.append(y)
            else:
                x_train.append(x.reshape(x.shape[0], x.shape[1], 1))
                y_train.append(y)
    print("Split: x = {} / {}".format(len(x_train), len(x_test)))
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)    
    cls_type.fit(x=x_train, y=y_train, batch_size=10, epochs=epoch)
    confusion = np.zeros((4,4))
    for x, y in zip(x_test, y_test):
            _y = np.argmax(cls_type.predict(x.reshape(1, x.shape[0], x.shape[1], 1)), axis=1)[0]
            confusion[y][_y] += 1
    np.savetxt('{}/confusion_type.csv'.format(output_folder), confusion)
    accuracy = np.sum(confusion * np.eye(4)) / np.sum(confusion)
    log.info(accuracy)
    log.info(confusion)
    cls_type.save('{}/type.h5'.format(output_folder))
    plot_confusion_matrix(confusion, ["noise", "echo", "burst", "whistle"], 'Type Classification')
    plt.savefig('{}/confusion_type.png'.format(output_folder))
    plt.close()


def train_silence(version_tag, input_folder, output_folder, params, encoder_file, batch, epoch, latent, freeze, transfer=True):
    """
    Train a silence dectector on top of an encoder

    :param version_tag: basically the model name
    :param input_folder: the folder with the training data
    :param output_folder: the folder to save the model
    :param params: window parameters
    :param encoder_file: a saved encoder
    :param batch: batch size
    :param epochs: number of training epochs
    """
    log.info("Training Silence Detector: {} {}".format(version_tag, epoch))
    if transfer:
        enc = load_model(encoder_file)
    else:
        _, enc = auto_encoder(
            (params.spec_win, params.n_fft_bins, 1), latent
        )    
    cls_sil = classifier(enc, 1, freeze)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for (x, y, f, _, _) in dataset(input_folder, params, sil, True):
        if np.random.uniform() > 0.6:
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x.reshape(x.shape[0], x.shape[1], 1))
            y_train.append(y)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train) 
    cls_sil.fit(x=x_train, y=y_train, batch_size=5, epochs=epoch)
    cls_sil.save('{}/sil.h5'.format(output_folder))

    confusion = np.zeros((2,2))
    for x, y in zip(x_test, y_test):
        y = int(y)
        _y = int(np.round(cls_sil.predict(x.reshape(1, x.shape[0], x.shape[1], 1))[0]))
        confusion[y][_y] += 1
    np.savetxt('{}/confusion.csv'.format(output_folder), confusion)
    accuracy = np.sum(confusion * np.eye(2)) / np.sum(confusion)
    log.info(accuracy)
    log.info(confusion)
    plot_confusion_matrix(confusion, ["noise", "dolphin"], 'Noise Classification')
    plt.savefig('{}/confusion.png'.format(output_folder))
    plt.close()

    
def train_auto_encoder(version_tag, input_folder, output_folder, params, latent, batch, epochs):
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
    """
    log.info("Training Auto Encoder: {}".format(version_tag))
    ae, enc = auto_encoder(
        (params.spec_win, params.n_fft_bins, 1), latent
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
    enc.save('{}/encoder.h5'.format(output_folder))
    ae.save('{}/auto_encoder.h5'.format(output_folder))


def evaluate_encoder(version_tag, input_folder, output_folder, encoder_file, params, k):
    """
    Evaluate an encoder for feature embedding

    :param version_tag: basically the model name
    :param input_folder: the folder with the data to embed
    :param output_folder: the folder to save the plots
    :param encoder_file: a saved encoder
    :param params: window parameters
    :param k: number of clusters
    """
    log.info("Evaluate Encoder: {}".format(version_tag))
    enc = load_model(encoder_file)
    visualize_2dfilters(output_folder, enc, [1], n_rows = 8)    
    x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for (x,_,_,_,_) in dataset(
        input_folder, params, no_label, False
    ) if np.random.uniform() < 0.1])
    log.info(x.shape)
    h = enc.predict(x)
    visualize_embedding("{}/embeddings.png".format(output_folder), h, x, k)


def test_reconstruction(folder, out, params):
    """
    Reconstruct 100 examples using the auto encoder
    """
    ae = load_model('{}/auto_encoder.h5'.format(out))
    gen = dataset(folder, params, no_label, True)
    i = 0
    plt.figure(figsize=(40, 40))
    for (x, _, f, _, _) in gen:
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


def write_audio(out, cluster_id, instances_clusters, grouped_by_cluster, min_support, max_support):
    """
    Write clusters as audio
    
    :param cluster_id: id to write
    :param instances_clusters: number of instances per cluster
    :param grouped_by_cluster: dict[clusters][filename][start, stop]
    :param min_support: minimum number of instances in cluster
    :param max_support: maximum number of instances in cluster
    """
    if instances_clusters[cluster_id] >= min_support and instances_clusters[cluster_id] < max_support:
        log.info("Audio result for cluster: {} {}".format(cluster_id, instances_clusters[cluster_id]))
        audio_bank = AudioSnippetCollection("{}/seq_cluster_{}.wav".format(out, cluster_id))
        for f, snippets in grouped_by_cluster[cluster_id].items():
            log.info("Cluster: {}, {}, {}".format(cluster_id, f, len(snippets)))
            for audio_snippet in audio_regions(f, snippets):                  
                audio_bank.write(audio_snippet)
        audio_bank.close()
        log.info("Done: {}".format(cluster_id))


def sequence_clustering(inp, out, embedder, min_support=1, n_writers=10, max_instances=None):    
    """
    Hierarchical cluster connected regions of whistles and bursts
    """
    log.info("Sequence Clustering")
    for filename in tf.io.gfile.listdir(inp):
        if filename.endswith('.ogg') or filename.endswith('.wav') or filename.endswith('.aiff'):
            name = filename.replace(".wav", "")
            name = name.replace(".ogg", "")            
            name = name.replace(".aiff", "")            

            in_path  = "{}/{}".format(inp, filename)
            out_path = "{}/embedding_{}.csv".format(out, name)
            log.info("\t {}".format(in_path))
            if not os.path.isfile(out_path):
                inducer = TypeExtraction.from_audiofile(in_path, embedder)
                inducer.save(out_path, append=True)
    
    clusters = hierarchical_clustering(out, max_instances=max_instances)
    grouped_by_filename = {}
    grouped_by_cluster  = {}
    i = 0
    k = 0
    for (start, stop, f, t, c) in clusters:
        if c not in grouped_by_cluster:
            grouped_by_cluster[c] = {}
        if f not in grouped_by_cluster[c]:
            grouped_by_cluster[c][f] = []
        grouped_by_cluster[c][f].append((start, stop))
        if f not in grouped_by_filename:
            grouped_by_filename[f] = []
        grouped_by_filename[f].append((start, stop, c, t, i))
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
    with mp.Pool(processes=n_writers) as pool:
        pool.starmap(write_audio, ((out, cluster_id, instances_clusters, grouped_by_cluster, min_support, 500) for cluster_id in range(0, k)))
    log.info('Done Writing')
    
    for f, regions in grouped_by_filename.items():
        filename = f.split(".")[0].split("/")[-1]
        log_path = "{}/seq_clustering_log_{}.csv".format(out, filename)
        log.info("writing: {}".format(log_path))
        with open(log_path, "w") as fp:
            fp.write("start,stop,file,cluster,type,region_id\n")
            for start, stop, c, t, i in regions:
                fp.write("{},{},{},{},{},{}\n".format(start, stop, f, c, t, i))
    log.info('Done Logs')

        
def generate_dataset(work_folder, annotations, out):
    """
    Generate an annotated dataset from clustering result

    :param work_folder: work folder
    :param annotations: annotation file
    :param out: output folder
    """
    log.info("Generate Dataset")
    i = 0
    for cluster, d in annotate_clustering(work_folder, annotations).items():
        log.info("\t\t{}".format(cluster))
        for filename, regions in d.items():
            log.info("\t\t{}".format(filename))
            r = [(start, stop) for start, stop, _ in regions]
            a = [a for _, _, a in regions]
            for annotation, audio_snippet in zip(a, audio_regions(filename, r)):
                fp = "{}/{}_clustering_{}.wav".format(out, annotation, i)
                log.info("\t\t{}".format(fp))
                audio_bank = AudioSnippetCollection(fp)
                audio_bank.write(audio_snippet)
                audio_bank.close()
                i += 1

                
def header():
    return """
    =================================================================
    Dolphin Machine Learning Pipeline
                
    usage for training:      python ml_pipeline/pipeline.py train config/default_config.yaml 
    usage for induction:     python ml_pipeline/pipeline.py induction config/induction_config.yaml 
    usage for annotation:    python ml_pipeline/pipeline.py annotate config/annotation.yaml
    usage for word spotting: python ml_pipeline/pipeline.py simplified config/word_spotting.yaml
    usage for auto tuning:   python ml_pipeline/pipeline.py autotune config/auto_tuning.yaml

    by Daniel Kyu Hwa Kohlsdorf
    =================================================================
    """
    
    
if __name__== "__main__":
    log.info(header())
    if len(sys.argv) == 3 and sys.argv[1] == 'train':
        c = yaml.load(open(sys.argv[2]))
        log.info("Parameters: {}".format(c))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        latent       = c['latent']
        batch        = c['batch']
        version      = c['version']
        epochs       = c['epochs']
        epochs_sup   = c['epochs_sup']
        viz_k        = c['viz_k']
        silence      = c['sil']
        type_class   = c['type_class']
        unsupervised = c['unsupervised']
        reconstruct  = c['reconstruct'] 
        output       = c['output']
        transfer     = c['transfer']
        freeze       = c['freeze'] 
        train_auto_encoder(version, unsupervised, output, params, latent, batch, epochs)
        evaluate_encoder(version, unsupervised, output, "{}/encoder.h5".format(output), params, viz_k)
        train_silence(version, silence, output, params, "{}/encoder.h5".format(output), batch, epochs_sup, latent, freeze, transfer)
        train_type(version, type_class, output, params, "{}/encoder.h5".format(output), batch, epochs_sup, latent, freeze, transfer)
        test_reconstruction(reconstruct, output, params)
    elif len(sys.argv) == 3 and sys.argv[1] == 'induction':
        c = yaml.load(open(sys.argv[2]))
        log.info("Parameters: {}".format(c))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        inp          = c['input']
        output       = c['output'] 
        enc_path     = c['encoding']
        enc             = load_model("{}/encoder.h5".format(output))
        silence         = load_model("{}/sil.h5".format(output))
        type_classifier = load_model("{}/type.h5".format(output))
        embedder        = SequenceEmbedder(enc, params, silence, type_classifier)
        sequence_clustering(inp, output, embedder)
        clustering_usage(output)
        log.info('Done !!!')
    elif len(sys.argv) == 3 and sys.argv[1] == 'annotate':        
        c = yaml.load(open(sys.argv[2]))
        work_folder  = c['work_folder']
        annotations  = c['annotations'] 
        out          = c['out']
        generate_dataset(work_folder, annotations, out)
    elif len(sys.argv) == 3 and sys.argv[1] == 'simplified':        
        c = yaml.load(open(sys.argv[2]))
        print("Parameters: {}".format(c))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        latent       = c['latent']
        batch        = c['batch']
        version      = c['version']
        epochs       = c['epochs']
        epochs_sup   = c['epochs_sup']
        viz_k        = c['viz_k']
        unsupervised = c['unsupervised']
        reconstruct  = unsupervised
        output       = c['output']
        train_auto_encoder(version, unsupervised, output, params, latent, batch, epochs)
        evaluate_encoder(version, unsupervised, output, "{}/encoder.h5".format(output), params, viz_k)
        test_reconstruction(reconstruct, output, params)
        enc          = load_model("{}/encoder.h5".format(output))
        embedder     = SequenceEmbedder(enc, params)
        sequence_clustering(unsupervised, output, embedder)
        clustering_usage(output)
    elif len(sys.argv) == 3 and sys.argv[1] == 'autotune':        
        c = yaml.load(open(sys.argv[2]))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        inputs       = c['inputs']
        output       = c['output']
        enc             = load_model("{}/encoder.h5".format(output))
        embedder        = SequenceEmbedder(enc, params)
        sequence_clustering(inputs, output, embedder)
