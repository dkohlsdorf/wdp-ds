import numpy as np
from numba import jit
import numpy as np
from collections import namedtuple


class Symbol(namedtuple('Symbol', 'id type start stop prob')):
    
    def __str__(self):
        return "{}:{}".format(self.id, self.type)

    def l1_merge(self, other):
        return self.id == other.id and self.type == other.type and self.stop > other.start
    
    
    def l2_merge(self, other):
        return self.stop > other.start

    def merge(self, other):
        return Symbol(self.id, self.type, self.start, other.stop, self.prob)
    
    
def regions(df, th):
    filtered = []
    for i, row in df.iterrows():
        if row['prob'] > th:
            filtered.append(Symbol(row['cluster'], row['knn'], row['start'], row['stop'], row['prob']))

    compressed = []
    current = filtered[0]
    for symbol in filtered[1:]:
        if current.l1_merge(symbol):
            current = current.merge(symbol)
        else:
            compressed.append(current)
            current = symbol

    regions = []
    current = []
    for symbol in compressed:
        if len(current) == 0 or current[-1].l2_merge(symbol):
            current.append(symbol)
        else:
            regions.append(current)
            current = []
    return regions


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
    elif type_a[0] == 'E' and type_b[0] == 'B' or type_a[0] == 'B' and type_b[0] == 'E':
        return 0.25
    elif type_a[0] == 'W' and type_b[0] == 'W':
        return 0.25
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


def score(a, b, gap):
    symbols_a = [s.id for s in a]
    symbols_b = [s.id for s in b]
    types_a   = [s.type for s in a]
    types_b   = [s.type for s in b]
    dp        = needleman_wunsch(symbols_a, symbols_b, types_a, types_b, gap)
    return dp[len(symbols_a),len(symbols_b)]
