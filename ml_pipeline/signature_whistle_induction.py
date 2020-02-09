import pickle
import numpy as np
import os 

from collections import namedtuple
from utils import *
from audio import *

class SignatureWhistleInduction(namedtuple("SignatureWhistleInduction", "embeddings starts stops types clusters files")):
    '''
    Induce Grammar around Signature Whistles

    1) Find all signature whistles 
    2) Walk outwards from the signature whistle    
    '''

    @classmethod
    def from_embedding(cls, folder, embedder):
        embeddings = []
        clusters   = []
        starts     = []
        stops      = []
        types      = []
        files      = [] 
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                path = "{}/{}".format(folder, filename)
                print("- Working on embedding {}".format(path))
                regions = embedder.embed(path)
                for x, f, start, stop, t, c in regions:
                    embeddings.append(x)
                    starts.append(start)
                    stops.append(stop)
                    types.append(t)
                    files.append(f)
                    clusters.append(c)
        return cls(embeddings, starts, stops, types, clusters, files) 