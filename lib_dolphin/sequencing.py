import numba
import numpy as np
import pandas as pd
import os
import sys
import math

from numba import jit
from collections import namedtuple

NGRAM_TYPE = 0


class Symbol(namedtuple('Symbol', 'id type start stop')):

    def eq(self, other, by_type):
        if by_type:
            return self.type == other.type
        else:
            return self.id == other.id

    def merge(self, other):
        return Symbol(self.id, self.type, self.start, other.stop)

    def string(self, by_type):
        if by_type:
            return "{}:{}:{}".format(self.type, self.start, self.stop)
        else:
            return "{}:{}:{}".format(self.id, self.start, self.stop)


Sequence = namedtuple('Sequence', 'symbols file offset')


class RuleMatch(namedtuple('RuleMatch', 'filename offset rule type start stop')):

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
            s = Symbol(row['cluster'], row['labels'], row['start'], row['stop'])
            symbols.append(s)
        sequence = Sequence(symbols, shotid, offset)
        sequences.append(sequence)
    return sequences


def ngram_stream(file, n):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        ngram         = []
        starts        = []
        stops         = []
        for symbol in row['string'].split(','):
            cmp = symbol.split(":")
            sid   = cmp[0]
            start = cmp[1]
            stop  = cmp[2] 

            ngram.append(sid)
            starts.append(start)
            stops.append(stop)
            if len(ngram) > n:
                ngram  = ngram[-n:]
                starts = starts[-n:]
                stops  = stops[-n:] 
            if len(ngram) == n:
                rule_match = RuleMatch(row['filename'], row['offset'], ngram, NGRAM_TYPE, starts[0], stops[-1])
                yield rule_match