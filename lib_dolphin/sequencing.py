import numpy as np
from numba import jit
import numpy as np


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
