import numpy as np 
import matplotlib.pyplot as plt
import random 
import sys

from sklearn.manifold.t_sne import TSNE
from tensorflow.keras.models import load_model
from convnet import data_gen

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x,y,img,ax=None,zoom=1):
    assert len(x) == len(y) and len(x) == len(img)
    if ax is None:
        ax = plt.gca()
    images = [OffsetImage(1.0 - img[i, :, :, 0].T, cmap='gray', zoom=zoom) for i in range(len(img))]
    x, y = np.atleast_1d(x, y)
    artists = []
    for i in range(0, len(x)):
        ab = AnnotationBbox(images[i], (x[i], y[i]), xycoords='data', frameon=True)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python model_analysis.py WIN FOLDER1 ... FOLDERN")
    else:
        ae = load_model('autoencoder.h5')
        encoder = load_model('encoder.h5')

        win = int(sys.argv[1])
        folders = sys.argv[2:]
        x = np.stack([x for x in data_gen(folders, win)])

        tsne = TSNE()
        h = encoder.predict(x)
        l = tsne.fit_transform(h)
        fig, ax = plt.subplots()
        imscatter([a[0] for a in l], [a[1] for a in l], x, ax, zoom=0.15)
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
