import numpy as np
import pandas as pd
import os
import sys
import math

from numba import jit
from collections import namedtuple


Symbol = namedtuple('Symbol', 'id type start stop') 

@jit(nopython=True)        
def max3(a, b, c):    
    x = a
    if b > x:
        x = b
    if c > x:
        x = c
    return x
        
@jit(nopython=True)
def similarity(symbol_a, type_a, symbol_b, type_b):
    if symbol_a == symbol_b:
        return 1.0
    elif type_a == type_b:
        return 0.5
    else:
        return -1.0

@jit(nopython=True)
def needleman_wunsch(symbols_a, symbols_b, types_a, types_b, gap = -1):        
    N = len(symbols_a)
    M = len(symbols_b)
    dp = np.zeros((N + 1, M + 1))
    dp[0,0] = 0.0
    dp[1:, 0] = -np.arange(1, N + 1)
    dp[0, 1:] = -np.arange(1, M + 1)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dp[i, j] = max3(
                dp[i - 1, j - 1] + similarity(symbols_a[i - 1], types_a[i - 1], symbols_b[j - 1], types_b[j - 1]),
                dp[i - 1, j] + gap, 
                dp[i, j - 1] + gap
            )
    return dp[N, M]

        
class Sequence(namedtuple('Sequence', 'symbols file offset')):
    
    @property
    def rle(self):
        compressed = []
        for symbol in self.symbols:
            if len(compressed) == 0 or symbol.id != compressed[-1].id or symbol.type != compressed[-1].type:
                compressed.append(symbol)
        return compressed
    
    def similarity(self, other, gap = -1):     
        a = self.rle
        b = other.rle
        return needleman_wunsch(np.array([symbol.id for symbol in a]),
                                np.array([symbol.id for symbol in b]),
                                np.array([symbol.type for symbol in a]),
                                np.array([symbol.type for symbol in b]))
        
        
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