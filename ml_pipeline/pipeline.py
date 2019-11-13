import sys
import yaml
import pickle


import time
import subprocess  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.models import load_model
from google.cloud import storage 

from audio import *
from feature_extractor import *
from classifier import *
from plots import *
from sequence_embedder import *


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
    n_processed = 0
    for epoch in range(epochs):
        batch = []
        for (x, y, _, _, _) in dataset(folder, params, lable, True):
            batch.append((x,y))
            total_loss = 0.0
            if len(batch) == batch_size:
                x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for x, _ in batch])
                if isinstance(batch[0][1], float):
                    y = np.array([y for _, y in batch])
                else:
                    y = np.stack([y.reshape(y.shape[0], y.shape[1], 1) for _, y in batch])
                loss = model.train_on_batch(x=x, y=y)
                if isinstance(loss, np.float):
                    total_loss += loss
                else:
                    total_loss += loss[0]
                batch = []
                if n_processed % 10 == 0:
                    print("#: {} EPOCH: {} LOSS: {}".format(n_processed, epoch, total_loss))
                    total_loss = 0.0
                n_processed += 1


def train_silence(version_tag, input_folder, output_folder, params, encoder_file, batch, epoch):
    '''
    Train a silence dectector on top of an encoder

    version_tag: basically the model name
    input_folder: the folder with the training data
    output_folder: the folder to save the model
    params: window parameters
    encoder_file: a saved encoder
    batch: batch size
    epochs: number of training epochs
    '''
    print("Training Auto Encoder: {}".format(version_tag))
    enc     = load_model(encoder_file)
    cls_sil = classifier(enc)
    train(input_folder, params, sil, cls_sil, batch, epochs)
    cls_sil.save('{}/sil.h5'.format(output_folder))
    

def test_silence(version_tag, input_folder, output_folder, params, sil_file):
    '''
    Evaluation of the accuracy as confusion matrix
    
    version_tag: basically the model name
    input_folder: the folder with the training data
    output_folder: the folder to save the model
    params: window parameters
    sil_file: saved silence detector
    '''
    print("Evaluate silence model {}".format(version_tag))
    silence = load_model(sil_file)
    confusion = np.zeros((2,2))
    for (x, y,_,_,_) in dataset(input_folder, params, sil, False):
        _y = int(np.round(silence.predict(x.reshape(1, x.shape[0], x.shape[1], 1))[0]))
        y = int(y)
        confusion[y][_y] += 1
    np.savetxt('{}/confusion.csv'.format(output_folder), confusion)
    accuracy = np.sum(confusion * np.eye(2)) / np.sum(confusion)
    print("Accuracy: {}".format(accuracy))
    print("Confusion")
    print(confusion)
    plot_confusion_matrix(confusion, ['not silence', 'silence'], 'Silence Classification')
    plt.savefig('{}/confusion.png'.format(output_folder))
    plt.close()
    
    
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
        input_folder, params, no_label, False
    )])
    visualize_embedding("{}/embeddings.png".format(output_folder), x, enc, k)


def run_embedder(seq_embedder, folder, output):
    '''
    Run sequence embedding on all files in a folder

    seq_embedder: a sequence embedder
    folder: folder containing wav files on google cloud
    '''
    print("Apply sequence embedder to {}".format(folder))
    devnull = open(os.devnull, 'w')
    client = storage.Client.from_service_account_json('secret.json') 
    bucket = client.get_bucket('wdp-data') 
    paths = [f.name for f in bucket.list_blobs(prefix=folder) if f.name.endswith('.m4a')] 
    regions = []
    for path in paths:
        print("- Working on embedding {}".format(path))
        start = time.time()
        with open("/tmp/audio.m4a", "wb") as file_obj: 
            blob = bucket.blob(path) 
            blob.download_to_file(file_obj)   
        subprocess.call(["ffmpeg", "-i", '/tmp/audio.m4a', '/tmp/audio.wav'], stdout=devnull, stderr=devnull) 
        for x in seq_embedder.embed('/tmp/audio.wav'):                
            regions.append(x)
        end = time.time()
        print("- Done on embedding n regions: {} took {} [sec]".format(len(regions, end - start)))
    pickle.dump(regions, open('{}/regions.p'.format(output), 'wb'))

    
def header():
    return """
    =================================================================
    Dolphin Machine Learning Pipeline
    
    Training: 
        - training unsupervised encoder / decoder
        - plot evaluations
        - training supervised silence detector
        - plot confusion matrix
    
    Run:
        - convert all files in input folder to spectrogram
        - extract silence detector to all windows
        - embed every window
        - cluster windows and write the results to csv (filename, start, stop, cluster)
    
    usage for training: python ml_pipeline/pipeline.py train default_config.yaml
    usage for testing:  python ml_pipeline/pipeline.py run application_config.yaml

    by Daniel Kyu Hwa Kohlsdorf
    =================================================================
    """
    
    
if __name__== "__main__":
    print(header())
    if  len(sys.argv) == 3 and sys.argv[1] == 'run':
        c = yaml.load(open(sys.argv[2]))
        print("Parameters: {}".format(c))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        k            = c['k']
        inp          = c['gs_input']
        output       = c['output']        
        enc          = load_model("{}/encoder.h5".format(output))
        silence      = load_model("{}/sil.h5".format(output))
        embedder     = SequenceEmbedder(enc, silence, params)
        run_embedder(embedder, inp, output)        
    elif len(sys.argv) == 3 and sys.argv[1] == 'train':
        c = yaml.load(open(sys.argv[2]))
        print("Parameters: {}".format(c))
        params       = WindowParams(c['spec_win'], c['spec_step'], c['fft_win'], c['fft_step'], c['highpass'])
        latent       = c['latent']
        batch        = c['batch']
        version      = c['version']
        epochs       = c['epochs']
        viz_k        = c['viz_k']
        silence      = c['sil']
        unsupervised = c['unsupervised']
        output       = c['output']        
        train_auto_encoder(version, unsupervised, output, params, latent, batch, epochs)
        evaluate_encoder(version, silence, output, "{}/encoder.h5".format(output), params, viz_k)
        train_silence(version, silence, output, params, "{}/encoder.h5".format(output), batch, epochs)
        test_silence(version, silence, output, params, "{}/sil.h5".format(output))        
