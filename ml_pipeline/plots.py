import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import *
from sklearn.manifold.t_sne import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

import logging
logging.basicConfig()
logplots = logging.getLogger('plots')
logplots.setLevel(logging.INFO)


COLORS = list(
    pd.read_csv('ml_pipeline/colors.txt', sep='\t', header=None)[1])

def get_color(i):
    return COLORS[i % len(COLORS)]


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
            frameon=True, bboxprops = dict(edgecolor=get_color(c[i])))
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
        plt.figure(figsize=(200, 10))
        w = encoder.layers[l].get_weights()[0]
        n = w.shape[-1]
        for i in range(0, n):
            frame = plt.subplot(n_rows, n//n_rows, i + 1)
            plt.imshow(w[:, :, 0, i].T, cmap='gray')
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
        plt.savefig("{}/filters{}.png".format(img_path, l))
        plt.close()
        
            
def visualize_embedding(img_path, embeddings, examples, figsize=(80, 60), zoom=0.15):
    """
    Plot the examples in the embedding space projected to 2D using
    t-sne

    :param img_path: path where the image is saved including the name
    :param embeddings: the embeddings to visualize and cluster
    :param examples: the associated spectrograms
    :param min_dist: distance threhsold 
    :param figsize: size of figure
    :param zoom: zoom the examples
    :returns: clusters    
    """
    
    tsne = TSNE()
    distances = []
    for _ in range(0, 1000):
        i = np.random.randint(0, len(embeddings))
        j = np.random.randint(0, len(embeddings))
        d = np.sqrt(np.sum(np.square(embeddings[i] - embeddings[j])))
        distances.append(d)
    th = np.percentile(distances, 50)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=th)
    c = clustering.fit_predict(embeddings)
    l = tsne.fit_transform(embeddings)
    f, ax = plt.subplots(figsize=figsize)
    imscatter([a[0] for a in l], [a[1] for a in l], c, examples, ax, zoom=zoom)
    plt.savefig(img_path)
    plt.close()
    return clustering, c
