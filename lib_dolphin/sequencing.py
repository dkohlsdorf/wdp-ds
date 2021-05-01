import numpy as np
from numba import jit
import numpy as np
from collections import namedtuple


FFT_STEP    = 128
RAW_AUDIO   = 5120
FIND_REJECT = 5 * FFT_STEP
LEN_REJECT  = RAW_AUDIO + FIND_REJECT


class Symbol(namedtuple('Symbol', 'id type start stop prob')):
    
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
        return Symbol(self.id, self.type, self.start, other.stop, self.prob)
    
    
def regions(df, th):
    filtered = []
    for i, row in df.iterrows():
        if row['prob'] > th and row['labels'] != 'NOISE':
            filtered.append(Symbol(row['cluster'], row['labels'], row['start'], row['stop'], row['prob']))
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
        return 1.0
    elif type_a[0] == 'E' and type_b[0] == 'B' or type_a[0] == 'B' and type_b[0] == 'E':
        return -1.0
    elif type_a[0] == 'W' and type_b[0] == 'W':
        return -1.0
    else:
        return -2.0

    
@jit(nopython=True)
def needleman_wunsch(symbols_a, symbols_b, types_a, types_b, gap, pam, normalize = True, w = 4):   
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
            if similarity[i, j] > 0:                
                distances[i, j] -= (similarity[i, j] - minsim) / (maxsim - minsim)
                distances[j, i] = distances[i, j] 
    return distances
