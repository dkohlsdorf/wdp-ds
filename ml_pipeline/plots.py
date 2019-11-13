import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AnchoredText
from matplotlib.patches import Rectangle

from sklearn.cluster import KMeans
from sklearn.manifold.t_sne import TSNE


COLORS = list(
    pd.read_csv('ml_pipeline/colors.txt', sep='\t', header=None)[1])

def plot_confusion_matrix(confusion, classes, title, cmap=plt.cm.Blues):
    '''
    Plot confusion matrix   
    
    confusion: confusion matrix
    classes: class label
    title: plot title    
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion.shape[1]),
           yticks=np.arange(confusion.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def imscatter(x, y, c, img, ax=None, zoom=1):
    '''
    Plots the images on a scatter plot with a colored frame

    x: x values of input
    y: y values of input
    c: colors for frames
    img: img for each point
    '''
    assert len(x) == len(y) and len(x) == len(img)
    if ax is None:
        ax = plt.gca()
    images = [OffsetImage(1.0 - img[i, :, :, 0].T, cmap='gray', zoom=zoom)
            for i in range(len(img))]
    x, y = np.atleast_1d(x, y)
    artists = []
    for i in range(0, len(x)):
        ab = AnnotationBbox(
            images[i],
            (x[i], y[i]),
            xycoords='data',
            frameon=True, bboxprops = dict(edgecolor=COLORS[c[i]]))
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def visualize_2dfilters(img_path, encoder, layers, n_rows = 8):
    '''
    Vizualize filter banks in encoding model
    
    img_path: path where the image is saved exluding the name
    encoder: a keras model with convolutions
    layers: layers we want to visualize
    n_rows: number of rows in the plot
    '''
    for l in layers:
        w = encoder.layers[l].get_weights()[0]
        n = w.shape[-1]
        for i in range(0, n):
            frame = plt.subplot(n_rows, n//n_rows, i + 1)
            plt.imshow(w[:, :, 0, i].T, cmap='gray')
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
        plt.savefig("{}/filters{}.png".format(img_path, l))
        plt.close()
        
            
def visualize_embedding(img_path, examples, encoder, k=240, figsize=(80, 60), zoom=0.15):
    '''
    Plot the examples in the embedding space projected to 2D using
    t-sne

    img_path: path where the image is saved including the name
    examples: the examples to embedd
    encoder: a keras model 
    k: number of clusters 
    figsize: size of figure
    zoom: zoom the examples
    '''
    km   = KMeans(n_clusters=k, max_iter=1024)
    tsne = TSNE()
    h = encoder.predict(examples)
    c = km.fit_predict(h)
    l = tsne.fit_transform(h)
    f, ax = plt.subplots(figsize=figsize)
    imscatter([a[0] for a in l], [a[1] for a in l], c, examples, ax, zoom=zoom)
    plt.savefig(img_path)
    plt.close()
