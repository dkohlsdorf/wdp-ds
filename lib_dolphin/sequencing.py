import numpy as np
import pandas as pd
import os
import sys
import math

from numba import jit
from collections import namedtuple


class Symbol(namedtuple('Symbol', 'id type start stop prob')):
    
    def __str__(self):
        return "{}:{}".format(self.id, self.type)

    def merge(self, other):
        return Symbol(self.id, self.type, self.start, other.stop, self.prob)
        
        
INSERT    = 1
DELETE    = 2
MATCH_ID  = 3
MATCH_T   = 4
SUBS      = 5

    
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
    return dp
        
class Sequence(namedtuple('Sequence', 'symbols file offset')):
    
    def rle(self, prob_th):
        filtered = [s for s in self.symbols if s.prob > prob_th]        
        if len(filtered) == 0:
            return []
        compressed = []
        current = filtered[0]
        for symbol in filtered[1:]:            
            if symbol.id != current.id or symbol.type != current.type:
                compressed.append(current)
                current = symbol
            else:
                current = current.merge(symbol)
        return compressed
    
    def ngrams(self, n, prob_th=0.75):
        compressed = self.rle(prob_th)
        if len(compressed) >= n:
            for i in range(n, len(compressed)):
                yield compressed[i - n: i]
    
    def similarity(self, other, gap = -1, prob_th=0.75):     
        a  = self.rle(prob_th)
        b  = other.rle(prob_th)
        if len(a) == 0 or len(b) == 0:
            return float('-inf'), []
        symbols_a = np.array([symbol.id for symbol in a])
        symbols_b = np.array([symbol.id for symbol in b])
        types_a   = np.array([symbol.type for symbol in a])
        types_b   = np.array([symbol.type for symbol in b])
        dp = needleman_wunsch(symbols_a, symbols_b, types_a, types_b, gap)
        i = len(a)
        j = len(b)
        path = []
        while i > 0 and j > 0:
            op     = DELETE
            min_dp = dp[i - 1, j] 
            if dp[i, j - 1] <= min_dp:
                op     = INSERT
                max_dp = dp[i, j - 1]
            if dp[i - 1, j - 1] <= min_dp:
                op     = MATCH_ID
                max_dp = dp[i - 1, j - 1]
            if op == MATCH_ID:
                if symbols_a[i - 1] == symbols_b[j - 1]:
                    op = MATCH_ID
                elif types_a[i - 1] == types_b[j - 1]:
                    op = MATCH_T
                else:
                    op = SUBS
            path.append([op, symbols_a[i - 1], symbols_b[j - 1], types_a[i - 1], types_b[j - 1], i - 1, j - 1])

            if op == DELETE:
                i -= 1
            elif op == INSERT:
                j -= 1
            else:
                i -= 1
                j -= 1            
        while i > 0:
            path.append([DELETE, symbols_a[i - 1], symbols_b[j], types_a[i - 1], types_b[j], i - 1, j])
            i -= 1
        while j > 0:
            path.append([INSERT, symbols_a[i], symbols_b[j - 1], types_a[i], types_b[j - 1], i, j - 1])
            j -= 1
        path.reverse()
        return dp[len(a), len(b)], path

    def __str__(self):
        return " ".join([str(symbol) for symbol in self.rle(0)])
        
        
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
            s = Symbol(row['cluster'], row['labels'], row['start'], row['stop'], row['prob'])
            symbols.append(s)
        sequence = Sequence(symbols, shotid, offset)
        sequences.append(sequence)
    return sequences