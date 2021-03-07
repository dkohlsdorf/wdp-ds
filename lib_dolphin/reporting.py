#
# TODO: Visualize reconstructions
# TODO: Visualize filters
# TODO: Export Cluster WavFiles for train and test
#

import matplotlib.pyplot as plt
import pandas as pd
import imageio

from lib_dolphin.audio import *

import json

from flask import Flask

COLORS = list(
    pd.read_csv('lib_dolphin/colors.txt', sep='\t', header=None)[1].apply(lambda x: x + "80")
)


def overlapping_regions(regions, use_cluster, anno):
    start = regions["start"]
    stop  = regions["stop"]
    n     = len(start)
    if use_cluster:
        annotation = regions['cluster']
    else:
        annotation = regions['labels'] 

    start_region = 0
    regions = []
    for i in range(1, n):
        if stop[i - 1] < start[i] or annotation[i - 1] != annotation[i]:            
            js = {
                "start"      : str(start[start_region]),
                "stop"       : str(start[i - 1]),
                "annotation" : str(annotation[i-1]),
                "color"      : COLORS[anno[annotation[i - 1]]]
            }
            regions.append(js)
            start_region = i
    return json.dumps(regions)


def savefig(wav, out, lo, hi, win, step):
    x  = raw(wav)
    s  = spectrogram(x, lo, hi, win, step)        
    imageio.imwrite(out, 1.0 - s.T)


def js_paint(sid, regions, use_cluster, anno):
    js = overlapping_regions(regions, use_cluster, anno)
    return "paint(\"{}\", {});\n".format(sid, js)


def html_paint(sid, folder, audio):
    filename = "{}/{}".format(folder, audio.split('/')[-1].replace('.wav', '.png'))
    savefig(audio, filename, 0, 256, 512, 128)
    return """
    <img id="{}_img" src="{}"/>
    <canvas id="{}_canvas"></canvas>   
    """.format(sid, filename, sid)


def template(sids, folder, audios, regions_csv, clusters):
    regions = []
    anno = {}
    cur = 0
    for csv in regions_csv:
        region = pd.read_csv(csv) 
        if clusters:
            annotation = region['cluster']
        else:
            annotation = region['labels']        
        for a in annotation:
            if a not in anno:
                anno[a] = cur
                cur += 1
        regions.append(region)
    js = []
    html = []
    
    for i in range(0, len(sids)):
        js.append(js_paint(sids[i], regions[i], clusters, anno))
        html.append(html_paint(sids[i], folder, audios[i]))
    js   = "\n".join(js)
    html = "\n".join(html)

    template = []
    for line in open('lib_dolphin/template.html'):
        if line.strip() == "HTML_HERE":
            template.append(html)
        elif line.strip() == "JS_HERE":
            template.append(js)
        else:
            template.append(line)
    return "\n".join(template)
