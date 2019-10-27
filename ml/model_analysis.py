import numpy as np 
import matplotlib.pyplot as plt
import random 
import sys
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold.t_sne import TSNE
from tensorflow.keras.models import load_model
from train_convnet_features import data_gen

from scipy.io import wavfile

from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AnchoredText
from matplotlib.patches import Rectangle

COLORS = list(pd.read_csv('ml/colors.txt', sep='\t', header=None)[1])

def imscatter(x,y,c,img,ax=None,zoom=1):
    assert len(x) == len(y) and len(x) == len(img)
    if ax is None:
        ax = plt.gca()
    images = [OffsetImage(1.0 - img[i, :, :, 0].T, cmap='gray', zoom=zoom) for i in range(len(img))]
    x, y = np.atleast_1d(x, y)
    artists = []
    for i in range(0, len(x)):
        ab = AnnotationBbox(images[i], (x[i], y[i]), xycoords='data', frameon=True, bboxprops =dict(edgecolor=COLORS[c[i]]))        
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python model_analysis.py WIN NOISE_TEST FOLDER1 ... FOLDERN")
    else:
        ae = load_model('autoencoder.h5')
        encoder = load_model('encoder.h5')
        noise_classifier = load_model('sil.h5')
        win = int(sys.argv[1])
        noise_test = sys.argv[2]
        folders = sys.argv[3:]

        x = np.stack([x for x in data_gen([noise_test], win)])
        y = noise_classifier.predict(x).flatten()
        not_noise = [sample == 0 for sample in y]
        regions = []
        for i in range(0, len(not_noise)):
            if not_noise[i]:
                start = i * 16 * 256 
                stop  = (i + 1) * 16 * 256 
                if len(regions) > 0: 
                    last  = regions[-1]
                    if start - last[1] < 48000 * 0.75:
                        start       = regions[-1][0]
                        regions[-1] = (start, stop)
                    else:
                        regions.append((start, stop))
                else:
                    regions.append((start, stop))

        regions = [(start, stop) for start, stop in regions if stop - start > (16 * 256)]
        print(regions)
        fs, data = wavfile.read('data/demo/06111101.wav')
        print(data.shape)
        audio = []
        for start, stop in regions:
            audio.extend(data[start:stop, 1])
            audio.extend(np.zeros(fs // 10))    
        audio = np.array(audio, dtype=data.dtype)
        print(audio.shape)    
        wavfile.write('not_sil.wav', fs, audio)
        x = np.stack([x for x, _ in data_gen(folders, win, lambda x: x.startswith('noise'))])
        y = [y for _, y in data_gen(folders, win, lambda x: x.startswith('noise'))]
        _y = noise_classifier.predict(x)
        sil_confusion = np.zeros((2, 2))
        for i in range(len(y)):
            sil_confusion[int(y[i])][int(_y[i][0])] += 1.0
        print(sil_confusion)

        km = KMeans(n_clusters=24, max_iter=1024)
        tsne = TSNE()
        h = encoder.predict(x)
        c = km.fit_predict(h)
        l = tsne.fit_transform(h)

        fig, ax = plt.subplots()
        imscatter([a[0] for a in l], [a[1] for a in l],c, x, ax, zoom=0.15)
        plt.show()

        w = encoder.layers[1].get_weights()[0]
        n = w.shape[3]
        for i in range(0, n):
            plt.subplot(8, n//8, i + 1)
            plt.imshow(w[:, :, 0, i].T, cmap='gray')
        plt.show()

        y = ae.predict(x)
        sample = np.arange(len(x))
        random.shuffle(sample)
        sample = sample[0:10]
        for i in sample:
            plt.subplot(1, 2,  1)
            plt.imshow(1.0 - x[i, :, :, 0].T, cmap='gray')
            plt.subplot(1, 2,  2)
            plt.imshow(1.0 - y[i, :, :, 0].T, cmap='gray')
            plt.show()
