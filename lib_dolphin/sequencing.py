import numba
import numpy as np
import pandas as pd
import os
import sys

from numba import jit
from collections import namedtuple

sys.setrecursionlimit(1500)

Symbol     = namedtuple('Symbol', 'id type')
Sequence   = namedtuple('Sequence', 'symbols file offset')

class RuleMatch(namedtuple('RuleMatch', 'filename offset rule')):

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
                rule_match = RuleMatch(row['filename'], row['offset'], ngram)
                yield rule_match


def rules_abl_stream(file, min_matches = 5):
    df = pd.read_csv(file)
    sequences = []
    for _, row in df.iterrows():
        sequences.append(np.array(row['string'].split(',')))
    n = len(sequences)
    rules = []
    closed = set([])
    for i in range(0, n):        
        for j in range(i + 1, n):
            distance, path = align(sequences[i], sequences[j])
            r = RegexpNode.from_alignment(path)
            n_matches = len([op for op, _, _ in path if op == MATCH])
            if n_matches >= min_matches and str(r) not in closed:
                rules.append(r)
                closed.add(str(r))
    print("#Rules = {}".format(len(rules)))
    for i, row in df.iterrows():
        strg = row['string'].split(',')
        print("Processing: {} / {} [{}]".format(i, len(df), len(strg)))
        for j, rule in enumerate(rules):
            if j % 100 == 0:
                print("\t\t... rule {}".format(j))
            if match(strg, rule):
                rule_match = RuleMatch(row['filename'], row['offset'], rule)
                yield rule_match


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


MATCH      = 0
DELETE     = 1
INSERT     = 2
SUBSTITUTE = 3 

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


def align(x, y):
    n  = len(x)
    m  = len(y)
    dp  = levenshtein(x, y)
    i = n
    j = m
    path = []
    while i > 0 and j > 0:
        op = DELETE
        min_dp = dp[i - 1, j] 
        if dp[i, j - 1] <= min_dp:
            op = INSERT
            max_dp = dp[i, j - 1]
        if dp[i - 1, j - 1] <= min_dp:
            op = MATCH
            max_dp = dp[i - 1, j - 1]
        if op == MATCH and x[i - 1] != y[j - 1]:
            op = SUBSTITUTE

        path.append([op, x[i - 1], y[j - 1]])

        if op == DELETE:
            i -= 1
        elif op == INSERT:
            j -= 1
        else:
            i -= 1
            j -= 1
    while i > 0:
        path.append([DELETE, x[i - 1], y[j]])
        i -= 1
    while j > 0:
        path.append([INSERT, x[i], y[j - 1]])
        j -= 1
    path.reverse()
    return dp[n, m], path


AND_SYMBOL    = -1
OR_SYMBOL     = -2
DONT_CARE     = -3
REPEAT        = -4


class RegexpNode:
    
    def __init__(self, symbol, children = []):
        self.symbol = symbol
        self.children = children

    @property
    def is_leaf(self):
        return len(self.children) == 0
    
    @property
    def repeat_any(self):
        if self.symbol == REPEAT:
            if len(self.children) == 1:
                return self.children[0].is_leaf and self.children[0].symbol == DONT_CARE
        return False
        
    @classmethod
    def from_alignment(cls, alignment, last = None):
        REPEAT_ANY = cls(REPEAT, [cls(DONT_CARE)])
        
        op, x, y = alignment[0]
        if op == MATCH:
            symbol = cls(x)
        elif op == SUBSTITUTE:
            symbol = cls(OR_SYMBOL, [cls(x), cls(y)])
        elif op == INSERT or op == DELETE:
            symbol = cls(REPEAT, [cls(DONT_CARE)])
        if len(alignment) == 1:
            if symbol.repeat_any and last.repeat_any:
                return None
            else:
                return symbol
        else:            
            if last is not None and symbol.repeat_any and last.repeat_any:
                next_symbol = RegexpNode.from_alignment(alignment[1:], last)
                return next_symbol
            next_symbol = RegexpNode.from_alignment(alignment[1:], symbol)
            if next_symbol is None:
                return symbol
            return cls(AND_SYMBOL, [symbol, next_symbol])
        

    def __str__(self):
        if self.is_leaf:
            if self.symbol == DONT_CARE:
                return "."
            return str(self.symbol) + " "
        elif self.symbol == AND_SYMBOL:
            return str(self.children[0]) + str(self.children[1])
        elif self.symbol == OR_SYMBOL:
            return "(" + str(self.children[0]) + "|" + str(self.children[1]) + ")"
        elif self.symbol == REPEAT:
            return "(" + str(self.children[0]) + ")" + "+ "


def match(string, regexp, depth = 0):
    if regexp.is_leaf and regexp.symbol != DONT_CARE:
        return len(string) == 1 and regexp.symbol == string[0]
    elif regexp.symbol == DONT_CARE:
        return len(string) == 1
    elif regexp.symbol == AND_SYMBOL:
        assert len(regexp.children) == 2
        result = False
        for i in range(0, len(string) + 1):
            result = result or match(string[0:i], regexp.children[0], depth + 1) and match(string[i: len(string)], regexp.children[1], depth + 1)
        return result
    elif regexp.symbol == OR_SYMBOL:
        assert len(regexp.children) == 2
        result = match(string, regexp.children[0], depth + 1) or match(string, regexp.children[1], depth + 1)
        return result
    elif regexp.symbol == REPEAT:
        if len(string) > 0:
            matches = -1
            for i in range(0, len(string) + 1):
                if match(string[0: i], regexp.children[0], depth + 1):
                    matches = i
            if matches < 0:
                return False
            return match(string[matches: len(string)], regexp, depth + 1)
        return True
