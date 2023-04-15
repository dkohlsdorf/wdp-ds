import numpy as np

from collections import namedtuple
from numba import jit

#from lib_dolphin.dtw import *
IndexRange = namedtuple("IndexingRange", "start stop")


class KWayIndexingTree:

    def __init__(self):
        self.selector = None
        self.left = None
        self.right = None

    def leaf(self):
        return self.left is None and self.right is None
    
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
def build_tree(data, make_selector, n_leaf=10):
    N = len(data)
    ids = np.arange(0, N)
    selections = np.zeros(N)
    root = KWayIndexingTree()
    ranges = [(IndexRange(0, N), root)]
    
    while len(ranges) > 0:
        i, node = ranges.pop()        
        if i.stop - i.start >= n_leaf:
            node.selector = make_selector(data, ids, i)
            for j in range(i.start, i.stop):
                selections[j] = node.selector(data, j)
            node.left = KWayIndexingTree()
            node.right = KWayIndexingTree()
            l, r = partition(ids, i, selections)
            ranges.append((l, node.left))
            ranges.append((r, node.right))
        else:
            node.selector = ids[i.start:i.stop]
    return root


        


# Test me ======================================================================

def test_selector(data, ids, idx):
    selector = lambda x, j: j % 2
    return selector 

x = [1,2,3,4,5,6]
y = [0,1,0,1,0,1]

idx1 = IndexRange(0, 6)

l, r = partition(x, idx1, y)
print(l, r, x, y)

print(x[l.start:l.stop], y[l.start:l.stop])
print(x[r.start:r.stop], y[r.start:r.stop])
    
root = build_tree(np.zeros((100, 10)), test_selector)

def explore(node, indent=''):
    if node.leaf():
        print(f"{indent}{node.selector}")
    else:
        print(f"{indent}node")
        explore(node.left, f"{indent} ")
        explore(node.right, f"{indent} ")

explore(root)        
