import numba
import numpy as np
import pandas as pd
import os
import sys
import math

from numba import jit
from collections import namedtuple

NGRAM_TYPE = 0

Symbol     = namedtuple('Symbol', 'id type')
Sequence   = namedtuple('Sequence', 'symbols starts stops file offset')

class RuleMatch(namedtuple('RuleMatch', 'filename offset rule type')):

    @property
    def position_id(self):
        return "filename:{}_offset:{}".format(self.filename, self.offset)

    @property
    def rule_id(self):
        return str(self.rule)
    

def extract_id(filename):
    fn = filename.replace(' ', '_')
    return fn.split('_')[0]


def extract_offset(filename):
    fn = filename.replace(' ', '_')
    return fn.split('_')[1].replace('.csv', '')


def extract_sequences(files):
    sequences = []
    for file, path in files:
        shotid        = extract_id(file)
        offset        = extract_offset(file)                
        df            = pd.read_csv(path)
        symbols = []
        starts  = []
        stops   = []
        for i, row in df.iterrows():
            starts = row['start']
            stops  = row['stop']
            s      = Symbol(row['cluster'], row['labels'])
            symbols.append(s)
            starts.append(starts)
            stops.append(stops)            
        sequence = Sequence(symbols, starts, stops, shotid, offset)
        sequences.append(sequence)
    return sequences


def ngram_stream(file, n):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        ngram = []
        for symbol in row['string'].split(','):
            ngram.append(symbol)
            if len(ngram) > n:
                ngram = ngram[-n:]
            if len(ngram) == n:
                rule_match = RuleMatch(row['filename'], row['offset'], ngram, NGRAM_TYPE)
                yield rule_match

                
def idf_extractor(rules, id_func):
    documents = set([])
    counts    = {}
    for rule in rules:
        if rule.rule_id not in counts:
            counts[rule.rule_id] = set()
        doc = id_func(rule)
        counts[rule.rule_id].add(doc)
        documents.add(doc)

    N   = len(documents)
    idf = {}
    for k, v in counts.items():
        idf[k] = math.log(N / len(v))
    return idf


def tfidf_extractor(rules, idf, id_func):
    counts = {}
    for rule in rules:
        doc = id_func(rule)
        if doc not in counts:
            counts[doc] = {}
        if rule.rule_id not in counts[doc]:
            counts[doc][rule.rule_id] = 0
        counts[doc][rule.rule_id] += 1
    tfidf = {}
    for doc, freq in counts.items():
        scaler = sum([c for _, c in freq.items()])
        frequencies = [(k, c / scaler * idf[k]) for k, c in freq.items()]
        frequencies = dict(frequencies)
        tfidf[doc]  = frequencies
    return tfidf
