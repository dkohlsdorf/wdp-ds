import numba
import numpy as np
import pandas as pd
import os

from numba import jit
from collections import namedtuple


Symbol   = namedtuple('Symbol', 'id type')
Sequence = namedtuple('Sequence', 'symbols file offset')


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
                yield row['filename'], row['offset'], ngram


def rules_abl(file):
    df = pd.read_csv(file)
    sequences = []
    for _, row in df.iterrows():
        sequences.append(np.array(row['string'].split(',')))
    print(sequences[10], sequences[11])
    print(levenshtein(sequences[10], sequences[11]))


@jit(nopython=True)
def err(x, y):
    if x == y:
        return 0
    else:
        return 1

    
@jit(nopython=True)
def min3(x, y, z):
    minimum = x
    if y < minimum:
        minimum = y
    if z < minimum:
        minimum = z
    return minimum


@jit(nopython=True)
def levenshtein(x, y):
    n  = len(x)
    m  = len(y)
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min3(
                dp[i - 1, j] + 1,
                dp[i - 1, j - 1] + err(x[i-1],y[j-1]),
                dp[i, j - 1] + 1
            )    
    return dp
