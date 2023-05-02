"""
Building A DTW Indexing Tree

Idea: 
  We select a random time series. Then we compute all distances to that series.
  We sample a second series proportional to the distances squared. We partition the
  data around the two sequences and recursively partition the resulting sub partitions.  

USAGE:
  X = np.random.uniform(size=(100, 10, 10)) * 100
  root = build_tree(X, dtw_selector)
  explore(root)        
  print(search(X[0], root))
"""

import numpy as np
import random

from collections import namedtuple
from numba import jit

from lib_dolphin.dtw import *
IndexRange = namedtuple("IndexingRange", "start stop")


class IndexingTree:

    def __init__(self):
        self.selector = None
        self.left = None
        self.right = None

    def leaf(self):
        return self.left is None and self.right is None

@jit
def search(x, tree):
    queue = [tree]
    while len(queue) > 0:
        node = queue.pop()        
        if node.leaf():
            return node.selector
        if node.selector(x) == 0:
            queue.append(node.left)
        else:
            queue.append(node.right)
            
    
@jit        
def partition(ids, idx, lr):
    i = idx.start - 1
    for j in range(idx.start, idx.stop):
        if lr[j] == 0:
            i += 1
            ids[i], ids[j] = ids[j], ids[i]
            lr[i], lr[j] = lr[j], lr[i]
    return IndexRange(idx.start, i + 1), IndexRange(i + 1, idx.stop)


@jit
def build_tree(data, make_selector, n_leaf=100):
    N = len(data)
    ids = np.arange(0, N)
    selections = np.zeros(N)
    root = IndexingTree()
    ranges = [(IndexRange(0, N), root)]
    
    while len(ranges) > 0:
        i, node = ranges.pop()        
        if i.stop - i.start >= n_leaf:
            node.selector = make_selector(data, ids, i)
            for j in range(i.start, i.stop):
                selections[j] = node.selector(data[ids[j]])
            node.left = IndexingTree()
            node.right = IndexingTree()
            l, r = partition(ids, i, selections)
            ranges.append((l, node.left))
            ranges.append((r, node.right))
        else:
            node.selector = ids[i.start:i.stop]
    return root


def dtw_selector(X, ids, idx):
    n = idx.stop - idx.start
    selected_idx = np.random.randint(n) + idx.start
    distances = np.array([dtw(X[selected_idx], X[i]) for i in range(idx.start, idx.stop) if i != selected_idx])
    distances = distances ** 2
    distances = distances / sum(distances)
    selected_idx_sample = random.choices([ids[i] for i in range(idx.start, idx.stop) if i != selected_idx], distances)[0]
    selector = lambda x: 0 if dtw(X[selected_idx], x) <= dtw(X[selected_idx_sample], x) else 1
    return selector


def explore(node, indent=''):
    if node.leaf():
        print(f"{indent}{node.selector}")
    else:
        print(f"{indent}node")
        explore(node.left, f"{indent} ")
        explore(node.right, f"{indent} ")
