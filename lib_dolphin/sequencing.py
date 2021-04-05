import numba
import numpy as np
import pandas as pd
import os
import sys

from numba import jit
from collections import namedtuple

NGRAM_TYPE = 0

Symbol     = namedtuple('Symbol', 'id type')
Sequence   = namedtuple('Sequence', 'symbols file offset')

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
        for i, row in df.iterrows():
            s = Symbol(row['cluster'], row['labels'])
            symbols.append(s)
        sequence = Sequence(symbols, shotid, offset)
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
