
import sys
import yaml
import pickle

import numpy as np
import subprocess  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.backend import set_learning_phase
import datetime

from tensorflow.keras.models import load_model
from feature_extractor import *
from classifier import *
from plots import *
from sequence_embedder import *
from audio_collection import *
from structured import *
from utils import * 


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


def train(folder, params, lable, model, batch_size=10, epochs=128, keep=lambda x: True):
    """
    Train the model for some epochs with a specific batch size

    :param data: a data iterator
    :param model: a keras model
    :param batch_size: size of the mini batch
    :param epochs: number of runs over the complete dataset
    :param keep: function from label to keep or not
    """
    n_processed = 0
    for epoch in range(epochs):
        batch = []
        for (x, y, _, _, _) in dataset(folder, params, lable, True):
            if keep(y):
                batch.append((x,y))
                total_loss = 0.0
                if len(batch) == batch_size:
                    x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for x, _ in batch])
                    if isinstance(batch[0][1], float):
                        y = np.array([y for _, y in batch])
                    else:
                        y = np.stack([y.reshape(y.shape[0], y.shape[1], 1) for _, y in batch])
                    loss = model.train_on_batch(x=x, y=y)
                    if isinstance(loss, np.float32):
                        total_loss += loss
                    else:
                        total_loss += loss[0]
                    batch = []
                    if n_processed % 10 == 0:
                        print("#: {} EPOCH: {} LOSS: {}".format(n_processed, epoch, total_loss))
                        total_loss = 0.0
                    n_processed += 1


def train_type(version_tag, input_folder, output_folder, params, encoder_file, batch, epoch, latent, freeze, transfer=True):
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
        if np.random.uniform() > 0.6:
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x.reshape(x.shape[0], x.shape[1], 1))
            y_train.append(y)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train) 
    cls_type.fit(x=x_train, y=y_train, batch_size=50, epochs=epoch)
    confusion = np.zeros((4,4))
    for x, y in zip(x_test, y_test):
            _y = np.argmax(cls_type.predict(x.reshape(1, x.shape[0], x.shape[1], 1)), axis=1)[0]
            confusion[y][_y] += 1
    np.savetxt('{}/confusion_type.csv'.format(output_folder), confusion)
    accuracy = np.sum(confusion * np.eye(4)) / np.sum(confusion)
    print(accuracy)
    print(confusion)
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
    print("Training Silence Detector: {} {}".format(version_tag, epoch))
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
    for (x, y, _, _, _) in dataset(input_folder, params, sil, True):
        if np.random.uniform() > 0.6:
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x.reshape(x.shape[0], x.shape[1], 1))
            y_train.append(y)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train) 
    cls_sil.fit(x=x_train, y=y_train, batch_size=50, epochs=epoch)
    cls_sil.save('{}/sil.h5'.format(output_folder))

    confusion = np.zeros((2,2))
    for x, y in zip(x_test, y_test):
        y = int(y)
        _y = int(np.round(cls_sil.predict(x.reshape(1, x.shape[0], x.shape[1], 1))[0]))
        confusion[y][_y] += 1
    np.savetxt('{}/confusion.csv'.format(output_folder), confusion)
    accuracy = np.sum(confusion * np.eye(2)) / np.sum(confusion)
    print(accuracy)
    print(confusion)
    plot_confusion_matrix(confusion, ["noise", "dolphin"], 'Noise Classification')
    plt.savefig('{}/confusion.png'.format(output_folder))
    plt.close()

    
def train_auto_encoder(version_tag, input_folder, output_folder, params, latent, batch, epochs):
    """
    Train an auto encoder for feature embedding

    :param version_tag: basically the model name
    :param input_folder: the folder with the training data
    :param output_folder: the folder to save the model
    :param params: window parameters
    :param latent: dimension of the latent space
    :param batch: batch size
    :param epochs: number of training epochs
    """
    print("Training Auto Encoder: {}".format(version_tag))
    ae, enc = auto_encoder(
        (params.spec_win, params.n_fft_bins, 1), latent
    )
    enc.summary()
    if os.path.exists('{}/encoder.h5'.format(output_folder)) and os.path.exists('{}/auto_encoder.h5'.format(output_folder)):
        print("\tloading previous weights")
        _enc = load_model('{}/encoder.h5'.format(output_folder))
        _enc.summary()
        _ae  = load_model('{}/auto_encoder.h5'.format(output_folder))
        enc.set_weights(_enc.get_weights())
        ae.set_weights(_ae.get_weights())
    w_before = enc.layers[1].get_weights()[0].flatten()
    train(input_folder, params, auto_encode, ae, batch, epochs)
    w_after = enc.layers[1].get_weights()[0].flatten()
    print("DELTA W:", np.sum(np.square(w_before - w_after)))
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
    print("Evaluate Encoder: {}".format(version_tag))
    enc = load_model(encoder_file)
    visualize_2dfilters(output_folder, enc, [1], n_rows = 8)    
    x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for (x,_,_,_,_) in dataset(
        input_folder, params, no_label, False
    )])
    h = enc.predict(x)
    visualize_embedding("{}/embeddings.png".format(output_folder), h, x, k)


def test_reconstruction(folder, out, params):
    '''
    Reconstruct 100 examples using the auto encoder
    '''
    ae = load_model('{}/auto_encoder.h5'.format(out))
    gen = pipe.dataset(folder, params, no_label, True)
    i = 0
    plt.figure(figsize=(40, 40))
    for (x, _, f, _, _) in gen:
        name = f.split('/')[-1]
        plt.subplot(10, 10, i + 1)
        plt.axis('off')
        plt.imshow(1.0 - ae.predict(x.reshape(1, 128, 256, 1))[0, :, :, 0].T, cmap='gray')
        i += 1
        if i % 10 == 0:        
            print(i)
        if i == 100:
            break
    plt.savefig('{}/reconstructions.png'.format(out))
    plt.close()


def sequence_clustering(inp, out, embedder, support=3):    
    print("Sequence Clustering")
    for filename in os.listdir(inp):
        if filename.endswith('.wav'):
            name = filename.replace(".wav", "")
            in_path  = "{}/{}".format(inp, filename)
            out_path = "{}/embedding_{}.csv".format(out, name)
            print("\t {}".format(in_path))
            if not os.path.isfile(out_path):
                inducer = TypeExtraction.from_audiofile(in_path, embedder)
                inducer.save(out_path, append=True)
    print("\n clustering:")
    clusters = [x for x in hierarchical_clustering(out)]            
    grouped_by_filename = {}
    # instance id
    for i, (start, stop, f, c) in enumerate(clusters):
        if f not in grouped_by_filename:
            grouped_by_filename[f] = []
        grouped_by_filename[f].append((start, stop, c, i))

    k = max([c for _, _, _, c in clusters]) + 1
    instances_clusters = np.zeros(k)
    for _, _, _, c in clusters:
        instances_clusters[c] += 1
    
    for cluster_id in range(0, k):
        if instances_clusters[cluster_id] > support:
            audio_bank = AudioSnippetCollection("{}/seq_cluster_{}.wav".format(out, cluster_id))
            for f, regions in grouped_by_filename.items():
                filename = f.split(".")[0].split("/")[-1]
                log_path = "{}/seq_clustering_log_{}.csv".format(out, filename)
                #instance id
                with open(log_path, "w") as fp:
                    for start, stop, c, i in regions:
                        fp.write("{},{},{},{},{}\n".format(start, stop, f, c, i))                
                snippets        = [(start, stop, f) for start, stop, _, _ in regions]
                cluster_snippet = [c for _, _, c,_ in regions] 
                for audio_snippet, c in zip(audio_snippets(snippets), cluster_snippet):
                    if c == cluster_id:
                        audio_bank.write(audio_snippet)
            audio_bank.close()
    

    
def signature_whistles(inp, out, embedder):
    for filename in os.listdir(inp):
        if filename.endswith('.wav'):
            name = filename.replace(".wav", "")
            in_path  = "{}/{}".format(inp, filename)
            out_path = "{}/embedding_{}.csv".format(out, name)
            log_path = "{}/signature_log_{}.csv".format(out, name)
            if not os.path.isfile(out_path):
                inducer = TypeExtraction.from_audiofile(in_path, embedder)
                inducer.save(out_path, append=True)
            snippets   = [(start, stop, f) for start, stop, _, f in signature_whistle_detector(out_path)]
            if len(snippets) > 0:
                audio_bank = AudioSnippetCollection("{}/signatures_{}.wav".format(out, name))
                for audio_snippet in audio_snippets(snippets):
                    audio_bank.write(audio_snippet)
                audio_bank.close()
                with open(log_path, "w") as fp:
                    for start, stop, dist, f in signature_whistle_detector(out_path):
                        fp.write("{},{},{},{}\n".format(start, stop, dist, f))
                        print("{} - {} {} {}".format(
                            str(datetime.timedelta(seconds=start/48000)),
                            str(datetime.timedelta(seconds=stop/48000)), dist, f))        
                
        
def header():
    return """
    =================================================================
    Dolphin Machine Learning Pipeline
                
    usage for training:  python ml_pipeline/pipeline.py train config/default_config.yaml
    usage for induction: python ml_pipeline/pipeline.py induction config/induction_config.yaml
    
    by Daniel Kyu Hwa Kohlsdorf
    =================================================================
    """
    
    
if __name__== "__main__":
    print(header())
    if len(sys.argv) == 3 and sys.argv[1] == 'train':
        c = yaml.load(open(sys.argv[2]))
        print("Parameters: {}".format(c))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        latent       = c['latent']
        batch        = c['batch']
        version      = c['version']
        epochs       = c['epochs']
        epochs_sup   = c['epochs_sup']
        viz_k        = c['viz_k']
        silence      = c['sil']
        unsupervised = c['unsupervised']
        reconstruct  = c['reconstruct'] 
        output       = c['output']
        transfer     = c['transfer']
        freeze       = c['freeze'] 
        train_auto_encoder(version, unsupervised, output, params, latent, batch, epochs)
        evaluate_encoder(version, unsupervised, output, "{}/encoder.h5".format(output), params, viz_k)
        train_silence(version, silence, output, params, "{}/encoder.h5".format(output), batch, epochs_sup, latent, transfer)
        train_type(version, type_class, output, params, "{}/encoder.h5".format(output), batch, epochs_sup, latent, transfer)
        test_reconstruction(reconstruct, output, params)
    elif len(sys.argv) == 3 and sys.argv[1] == 'induction':
        c = yaml.load(open(sys.argv[2]))
        print("Parameters: {}".format(c))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        inp          = c['input']
        output       = c['output'] 
        enc             = load_model("{}/encoder.h5".format(output))
        silence         = load_model("{}/sil.h5".format(output))
        type_classifier = load_model("{}/type.h5".format(output))
        km              = unpickle("{}/km.p".format(output))
        embedder        = SequenceEmbedder(enc, silence, type_classifier, km, params)
        signature_whistles(inp, output, embedder) 
        sequence_clustering(inp, output, embedder)
