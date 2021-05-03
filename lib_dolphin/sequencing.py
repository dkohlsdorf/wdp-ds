import numpy as np
import pandas as pd

from collections import namedtuple
from numba import jit

import warnings
warnings.filterwarnings('ignore')


MAX_BAND    = 1000
FFT_STEP    = 128
RAW_AUDIO   = 5120
FIND_REJECT = 5 * FFT_STEP
LEN_REJECT  = RAW_AUDIO + FIND_REJECT


class Symbol(namedtuple('Symbol', 'id type start stop')):
    
    def __str__(self):
        return "{}:{}".format(self.id, self.type)

    def l1_merge(self, other):
        return self.id == other.id and self.type == other.type and self.stop > other.start
    
    def l2_merge(self, other):
        clicks  = other.type[0] == 'E' and self.type[0] == 'B' or other.type[0] == 'B' and self.type[0] == 'E'
        whistle = other.type[0] == 'W' and self.type[0] == 'W'    
        types   = other.type == self.type 
        overlap = (self.stop + FIND_REJECT) > other.start
        return overlap and (clicks or whistle or types)

    def merge(self, other):
        return Symbol(self.id, self.type, self.start, other.stop)
    
    
def regions(df, label_col = 'smooth'):
    filtered = []    
    for i, row in df.iterrows():        
        if row[label_col] >= 0:
            filtered.append(Symbol(row[label_col], row['knn'], row['start'], row['stop']))
    if len(filtered) == 0:
        return []
    compressed = []
    current = filtered[0]
    for symbol in filtered[1:]:
        if current.l1_merge(symbol):
            current = current.merge(symbol)
        else:
            compressed.append(current)
            current = symbol
    compressed.append(current)
    regions = []
    current = []
    for symbol in compressed:
        if len(current) == 0 or current[-1].l2_merge(symbol):            
            current.append(symbol)
        else:
            if current[-1].stop - current[0].start > LEN_REJECT:
                regions.append(current)
            current = []
    if len(current) > 0:
        if current[-1].stop - current[0].start > LEN_REJECT:
            regions.append(current)
    return regions


@jit(nopython=True)        
def pam(c, x):
    n  = len(c) 
    nc = max(set(c)) + 1
    inter_class = np.zeros((nc, nc))
    counts = np.zeros((nc, nc)) 
    for i in range(0, nc):
        for j in range(i, nc):
            for a in range(0, n):
                for b in range(a + 1, n):
                    if c[a] == i and c[b] == j:
                        dist = np.sqrt(np.sum(np.square(x[a] - x[b])))
                        if i != j:
                            inter_class[i, j] -= dist
                        else:
                            inter_class[i, j] += 1.0 / dist
                        inter_class[j, i] = inter_class[i, j]
                        counts[i, j]      += 1
                        counts[j, i]      = counts[i, j]
    
    inter_class /= counts     
    return inter_class


@jit(nopython=True)        
def max3(a, b, c):    
    x = a
    if b > x:
        x = b
    if c > x:
        x = c
    return x


@jit(nopython=True)        
def imax(a, b):    
    x = a
    if b > x:
        x = b
    return x


@jit(nopython=True)        
def imin(a, b):    
    x = a
    if b < x:
        x = b
    return x


@jit(nopython=True)
def similarity(symbol_a, type_a, symbol_b, type_b):
    if symbol_a == symbol_b:
        return 2.0
    elif type_a == type_b:
        return -1.0
    elif type_a[0] == 'E' and type_b[0] == 'B' or type_a[0] == 'B' and type_b[0] == 'E':
        return -2.0
    elif type_a[0] == 'W' and type_b[0] == 'W':
        return -2.0
    else:
        return -3.0

    
@jit(nopython=True)
def needleman_wunsch(symbols_a, symbols_b, types_a, types_b, gap, pam, normalize = False, w = MAX_BAND):   
    N = len(symbols_a)    
    M = len(symbols_b)    
    w = imax(w, abs(N - M)) + 2
    
    dp = np.ones((N + 1, M + 1)) * -imax(N, M)
    dp[0,0] = 0.0    
    for i in range(1, N + 1):    
        for j in range(imax(1, i - w), imin(M + 1, i + w)):
            if pam is None:
                sim = similarity(symbols_a[i - 1], types_a[i - 1], symbols_b[j - 1], types_b[j - 1])
            else:
                sim = pam[symbols_a[i - 1], symbols_b[j - 1]]
            dp[i, j] = max3(
                dp[i - 1, j - 1] + sim,
                dp[i - 1, j] + gap, 
                dp[i, j - 1] + gap
            )
    if normalize:
        return dp / (N + M)
    else:
        return dp


def score(a, b, gap, pam):
    symbols_a = np.array([s.id for s in a])
    symbols_b = np.array([s.id for s in b])
    types_a   = np.array([s.type for s in a])
    types_b   = np.array([s.type for s in b])
    dp        = needleman_wunsch(symbols_a, symbols_b, types_a, types_b, gap, pam)
    return dp[len(symbols_a),len(symbols_b)]


def distances(sequences, gap, pam = None, only_positive=True):
    n = len(sequences)
    similarity = np.zeros((n, n))
    distances  = np.ones((n, n))
    for i in range(0, n):
        if i % 50 == 0:
            print("Processing: {}".format(i))
        for j in range(i + 1, n):
            a = sequences[i]
            b = sequences[j]
            s = score(a, b, gap, pam)
            similarity[i, j] = s
            
    scores = similarity.flatten()
    if only_positive:
        scores = scores[scores > 0.0]
    minsim = np.min(scores) 
    maxsim = np.max(scores)
    print("Min / Max: {} / {}".format(minsim, maxsim))
    for i in range(0, n):
        for j in range(i + 1, n):
            if similarity[i, j] > 0 or not only_positive:
                distances[i, j] -= (similarity[i, j] - minsim) / (maxsim - minsim)
                distances[j, i] = distances[i, j] 
    return distances



class Decodable(namedtuple('Decodeable', 'start stop density')):
    
    def overlap(self, other):
        return self.stop > other.start

    def merge(self, other):
        return Decodable(self.start, other.stop, self.density + other.density)
    
    
def dense(list_dict, noise_p):
    classes = dict(list_dict)
    if noise_p > 0:
        classes[-1] = noise_p
    return classes


@jit
def viterbi_smoothing(x, n_clusters=22, p_same=0.5):
    N  = len(x.density)
    dp = np.ones((N, n_clusters + 1)) * float('-inf')
    bp = np.zeros((N, n_clusters + 1), dtype=np.int)
    
    start = np.argmax(dp[0])
    for i in range(0, n_clusters + 1):
        if i - 1 in x.density[0]:
            dp[0, i] = np.log(x.density[0][i - 1]) 
        bp[0, i] = start
    for t in range(1, N): 
        for i in range(0, n_clusters + 1):
            arg_max = 0
            max_val = float('-inf')
            for j in range(0, n_clusters + 1):
                if i == j:
                    ll = dp[t - 1, j] + np.log(p_same)
                else:
                    ll = dp[t - 1, j] + np.log(1.0 - p_same)
                if ll > max_val:
                    max_val = ll
                    arg_max = j
            bp[t, i] = arg_max
            if i - 1 in x.density[t]:
                dp[t, i] = max_val + np.log(x.density[t][i - 1])
            else:
                dp[t, i] = max_val + np.log(0.0)
    path = [np.argmax(dp[-1]) - 1]
    t = N - 2
    while t >= 0:
        path.append(bp[t, path[-1]] - 1)
        t -= 1
    return path


def err(x, y):
    errors = 0
    for i in range(0, int(min(len(x), len(y)))):
        if x[i] != y[i]:
            errors += 1
    return errors + np.abs(len(x) - len(y))


def smooth(decodables, before, df, filename):
    paths = []
    for d in decodables:
        p = viterbi_smoothing(d)
        p.reverse()
        paths = paths + p
    df['smooth']   = paths
    df['unsmooth'] = before

    print("{}: len(df) = {} / len(path) = {} / errors = {}".format(filename, len(paths), len(df), err(paths, before)))
    df = df[['unsmooth', 'smooth', 'start', 'stop', 'labels', 'knn', 'cluster', 'prob' , 'density']]
    df.to_csv(filename, index=False)
    
    
def decoded(df):
    before_smoothing = []
    decodables = []
    cur = None
    n = 0    
    for i, row in df.iterrows():
        if row['labels'] == 'NOISE':
            p_noise = row['prob']
        else:
            p_noise = 0
        
        labeling = list(dense(row['density'], p_noise).items())
        labeling.sort(key=lambda x: x[1])
        label = labeling[-1][0]
        before_smoothing.append(label)
        if cur == None:            
            cur = Decodable(row['start'], row['stop'], [dense(row['density'], p_noise)])
        else:
            x = Decodable(row['start'], row['stop'], [dense(row['density'], p_noise)])
            if cur.overlap(x):
                cur = cur.merge(x)
            else:                
                n += len(cur.density)
                decodables.append(cur)
                cur = x
        
    n += len(cur.density)
    decodables.append(cur)
    return decodables, before_smoothing
