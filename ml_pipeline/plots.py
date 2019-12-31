import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import KMeans
from sklearn.manifold.t_sne import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score


COLORS = list(
    pd.read_csv('ml_pipeline/colors.txt', sep='\t', header=None)[1])


def plot_confusion_matrix(confusion, classes, title, cmap=plt.cm.Blues):
    """
    Plot confusion matrix

    :param confusion: confusion matrix
    :param classes: class label
    :param title: plot title
    :param cmap: color map

    shamelessly stolen from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    fig, ax = plt.subplots()
    im = ax.imshow(confusion, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion.shape[1]),
           yticks=np.arange(confusion.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f'
    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, format(confusion[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def imscatter(x, y, c, img, ax=None, zoom=1):
    """
    Plots the images on a scatter plot with a colored frame

    :param x: x values of input
    :param y: y values of input
    :param c: colors for frames
    :param img: img for each point
    :param ax: plot axis
    :param zoom: image zoom
    """
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
    """
    Vizualize filter banks in encoding model

    :param img_path: path where the image is saved exluding the name
    :param encoder: a keras model with convolutions
    :param layers: layers we want to visualize
    :param n_rows: number of rows in the plot
    """
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
        
            
def visualize_embedding(img_path, embeddings, examples, k=240, figsize=(80, 60), zoom=0.15, sparse=False):
    """
    Plot the examples in the embedding space projected to 2D using
    t-sne

    :param img_path: path where the image is saved including the name
    :param embeddings: the embeddings to visualize and cluster
    :param examples: the associated spectrograms
    :param k: number of clusters
    :param figsize: size of figure
    :param zoom: zoom the examples
    :param sparse: sparsify clusters by shillouette 
    :returns: clusters    
    """
    km   = KMeans(n_clusters=k, max_iter=1024, n_jobs=-1)
    tsne = TSNE()
    c = km.fit_predict(embeddings)
    l = tsne.fit_transform(embeddings)
    if sparse:
        sample_silhouette_values = silhouette_samples(embeddings, c)
        th = np.percentile(sample_silhouette_values, 50)
        c = [cluster for cluster, shillouette in zip(c, sample_silhouette_values)   if shillouette > th]
        l = [latent for latent, shillouette in zip(l, sample_silhouette_values)     if shillouette > th]
        examples = np.stack([x for x, shillouette in zip(examples, sample_silhouette_values) if shillouette > th])
        ids = [i for i in range(0, len(sample_silhouette_values)) if sample_silhouette_values[i] > th]
        print("Shillouette TH: {} n_samples left {}".format(th, len(ids)))
    else:
        ids = [i for i in range(0, len(c))]
    f, ax = plt.subplots(figsize=figsize)
    imscatter([a[0] for a in l], [a[1] for a in l], c, examples, ax, zoom=zoom)
    plt.savefig(img_path)
    plt.close()
    return c, km, ids
