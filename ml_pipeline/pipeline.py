import sys
import yaml
import pickle

import numpy as np
import subprocess  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.backend import set_learning_phase

from tensorflow.keras.models import load_model
from feature_extractor import *
from classifier import *
from plots import *
from sequence_embedder import *
from generate_report import *
from audio_collection import *
from signature_whistle_induction import *
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


def run_embedder_fs(seq_embedder, folder, output, bucket_size = 1000):
    """
    Run sequence embedding on all files in a folder

    :param seq_embedder: a sequence embedder
    :param folder: folder containing wav files
    """
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            path = "{}/{}".format(folder, filename)
            print("- Working on embedding {}".format(path))
            regions = seq_embedder.embed(path)
            f = filename.replace('.wav', '.p')
            pickle.dump(regions, open('{}/regions_{}.p'.format(output, filename), 'wb'))
            print("- Done on embedding n regions: {}".format(len(regions)))
            

def run_embedder_gs(seq_embedder, folder, output, bucket_size = 1000):
    """
    Run sequence embedding on all files in a bucket from google cloud

    :param seq_embedder: a sequence embedder
    :param folder: folder containing wav files on google cloud
    """
    from google.cloud import storage
    print("Apply sequence embedder to {}".format(folder))
    cmp = folder.replace("gs://", "").split('/')
    bucket_path = cmp[0]
    folder = "/".join(cmp[1:])
    log = open('audio_log.txt', 'w')
    client = storage.Client.from_service_account_json('secret.json') 
    bucket = client.get_bucket(bucket_path)
    paths = [f.name for f in bucket.list_blobs(prefix=folder) if f.name.endswith('.m4a')] 
    for path in paths:
        print("- Working on embedding {}".format(path))
        with open("/tmp/audio.m4a", "wb") as file_obj: 
            blob = bucket.blob(path) 
            blob.download_to_file(file_obj)   
        subprocess.call(['ffmpeg', '-y', '-i', '/tmp/audio.m4a', '/tmp/audio.wav'], stdout=log, stderr=log) 
        regions = seq_embedder.embed('/tmp/audio.wav')
        f = path.split('/')[-1].replace('.wav', '.p')
        pickle.dump(regions, open('{}/regions_{}.p'.format(output, f), 'wb'))
        print("- Done on embedding n regions: {}".format(len(regions)))

        
def clustering_audio(embedding_folder, wav_folder, k, cloud=True):
    '''
    Write clusters into audio files
    
    :param embedding_folder: where the embeddings are coming from
    :param folder: folder containing wav files on google cloud
    :param k: number of clusters
    :param cloud: is the data coming from the cloud
    '''
    print("Dump Clustering Audio")
    # group clusters by filename
    grouped_by_filename = {}
    for line in open('{}/clusters.csv'.format(embedding_folder)):
        cmp      = line.split(',')
        cluster  = int(cmp[0])
        filename = cmp[1]
        start    = int(cmp[2])
        stop     = int(cmp[3])
        if filename not in grouped_by_filename:
            grouped_by_filename[filename] = [(cluster, start, stop)]
        else:
            grouped_by_filename[filename].append((cluster, start, stop))    
    if cloud: 
        from google.cloud import storage
        cmp = wav_folder.replace("gs://", "").split('/')
        bucket_path = cmp[0]
        path = "/".join(cmp[1:])
        client = storage.Client.from_service_account_json('secret.json') 
        bucket = client.get_bucket(bucket_path)                                
    audio_bank = [AudioSnippetCollection("{}/cluster_{}.wav".format(embedding_folder, i)) for i in range(0, k)]
    for filename in grouped_by_filename:
        regions  = [(start, stop) for (_, start, stop) in grouped_by_filename[filename]]
        clusters = [c             for (c, _, _) in grouped_by_filename[filename]]
        if cloud:
            fname = filename.replace(".p", "").replace("regions_", "")            
            log = open('audio_log.txt', 'w')
            with open("/tmp/audio.m4a", "wb") as file_obj: 
                print("- Process: {}{}".format(path, fname))
                blob = bucket.blob("{}{}".format(path, fname))
                blob.download_to_file(file_obj)   
            subprocess.call(['ffmpeg', '-y', '-i', '/tmp/audio.m4a', '/tmp/audio.wav'], stdout=log, stderr=log)             
            for i, region in enumerate(audio_regions('/tmp/audio.wav', regions)):
                cluster = clusters[i]
                audio_bank[cluster].write(region)                
        else:
            for region in spectrogram_regions(filename, regions):            
                cluster = clusters[i]
                audio_bank[cluster].write(region)
        
        
def evaluate_embedding(embedding_folder, wav_folder, params, k, p_keep = 1.0, cloud=True, sparsify=False):
    '''
    Evaluate a large embedding run

    :param embedding_folder: where the embeddings are coming from
    :param params: windowing parameters
    :param k: number of clusters
    :param p_keep: probability of keeping a sample
    :param cloud: is the data coming from the cloud
    :param sparsify: sparsify clusters by shillouette 
    '''
    print("Evaluate embeddings: Clustering and Scatter Plot Experiment")
    # group all embeddings found by file
    grouped_by_filename = {}
    for filename in os.listdir(embedding_folder):
        if filename.startswith('region') and filename.endswith('.p'):
            path = "{}/{}".format(embedding_folder, filename)
            embeddings = pickle.load(open(path, "rb"))
            for (x, f, start, stop, t, c) in embeddings:
                if not cloud:
                    filename = f
                if np.random.uniform() < p_keep:
                    if filename not in grouped_by_filename:
                        grouped_by_filename[filename] = [(x, start, stop)]
                    else:
                        grouped_by_filename[filename].append((x, start, stop))
    if cloud: 
        from google.cloud import storage
        cmp = wav_folder.replace("gs://", "").split('/')
        bucket_path = cmp[0]
        path = "/".join(cmp[1:])
        client = storage.Client.from_service_account_json('secret.json') 
        bucket = client.get_bucket(bucket_path)                        
    # collect all spectrograms, embeddings and regions
    spectrograms = []
    embeddings   = []
    all_regions  = []

    for filename in grouped_by_filename:
        regions = [(start, stop) for (_, start, stop) in grouped_by_filename[filename]]
        x       = [x for x, _, _ in  grouped_by_filename[filename]] 
        if cloud:
            fname = filename.replace(".p", "").replace("regions_", "")            
            log = open('audio_log.txt', 'w')
            with open("/tmp/audio.m4a", "wb") as file_obj: 
                print("- Process: {}{}".format(path, fname))
                blob = bucket.blob("{}{}".format(path, fname))
                blob.download_to_file(file_obj)   
            subprocess.call(['ffmpeg', '-y', '-i', '/tmp/audio.m4a', '/tmp/audio.wav'], stdout=log, stderr=log) 
            for region in spectrogram_regions('/tmp/audio.wav', params, regions):            
                spectrograms.append(region.reshape(region.shape[0], region.shape[1], 1))
            embeddings.extend(x)
            all_regions.extend([(filename, start, stop) for (_, start, stop) in grouped_by_filename[filename]])
        else:
            regions = [(start, stop) for (_, start, stop) in grouped_by_filename[filename]]
            x       = [x for x, _, _ in  grouped_by_filename[filename]] 
            for region in spectrogram_regions(filename, params, regions):            
                spectrograms.append(region.reshape(region.shape[0], region.shape[1], 1))
            embeddings.extend(x)
            all_regions.extend([(filename, start, stop) for (_, start, stop) in grouped_by_filename[filename]])            
    spectrograms = np.stack(spectrograms)
    embeddings   = np.stack(embeddings)
    # visualize the results and save the clustering
    clusters, km, ids = visualize_embedding("{}/embeddings_test.png".format(embedding_folder), embeddings, spectrograms, k, sparse=sparsify)
    pickle.dump(km, open("{}/km.p".format(embedding_folder), "wb"))
    with open("{}/clusters.csv".format(embedding_folder), "w") as fp:
        for i, x_id in enumerate(ids):
            filename, start, stop = all_regions[x_id]
            cluster = clusters[i]
            fp.write("{},{},{},{}\n".format(cluster, filename, start, stop))
    

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


def induction(inp, embedder):
    inducer = SignatureWhistleInduction.from_embedding(inp, embedder)
    n = len(inducer.starts)
    with open("induction_str.csv", "w") as fp:
        for i in range(n):
            print("{}\t{}\t{}\t{}\t{}".format(inducer.starts[i], inducer.stops[i], inducer.types[i], inducer.clusters[i],  ','.join(['%.5f' % f for f in inducer.embeddings[i]])))
            fp.write("{}\t{}\t{}\t{}\t{}\n".format(inducer.starts[i], inducer.stops[i], inducer.types[i], inducer.clusters[i],  ','.join(['%.5f' % f for f in inducer.embeddings[i]])))


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
        - plot the embeddings
    
    Report:
        - Compile a model report including
        - Confusion Matrix for silence detection
        - Convolutional Filters
        - Clustering Image
        - Write audio clusters
        
    usage for training:  python ml_pipeline/pipeline.py train config/default_config.yaml
    usage for testing:   python ml_pipeline/pipeline.py run config/application_config.yaml
    usage for report:    python ml_pipeline/pipeline.py report config/application_config.yaml
    usage for induction: python ml_pipeline/pipeline.py induction config/induction_config.yaml
    
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
        inp          = c['input']
        output       = c['output'] 
        enc          = load_model("{}/encoder.h5".format(output))
        silence      = load_model("{}/sil.h5".format(output))
        type_classifier = load_model("{}/type.h5".format(output))
        embedder     = SequenceEmbedder(enc, silence, type_classifier, params)
        if inp.startswith('gs://'):
            run_embedder_gs(embedder, inp, output)
            evaluate_embedding(output, inp, params, k, 0.25, True, True)
            clustering_audio(output, inp, k, True)            
        else:
            run_embedder_fs(embedder, inp, output)
            evaluate_embedding(output, inp, params, k, 0.25, False, True)
            clustering_audio(output, inp, k, True)
    elif len(sys.argv) == 3 and sys.argv[1] == 'report':
        c = yaml.load(open(sys.argv[2]))
        print("Parameters: {}".format(c))
        from_template('ml_pipeline/reporting_template.md', c)
    elif len(sys.argv) == 3 and sys.argv[1] == 'train':
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
        embedder     = SequenceEmbedder(enc, silence, type_classifier, km, params)
        induction(inp, embedder) 