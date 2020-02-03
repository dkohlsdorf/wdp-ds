from collections import namedtuple
from dtw import DTW

import unittest
import numpy as np


MATCH_CHAR = '.'
GAP_CHAR   = '_' 

GAP   = 1
MATCH = 0


def path_process(path):
    """
    Compute matches and gaps along an alignment path

    :param path: list(index)

    :returns: list (index, GAP:1 / MATCH:0)
    """
    result = [(0, MATCH)]
    n = len(path)
    for i in range(1, n):
        if path[i] == path[i - 1]:            
            result.append((i, GAP))
        else:
            result.append((i, MATCH))
    return result


def result_symbol_str(i, s, sequence, path):
    """
    Build an alignment symbol

    :param i: position in path
    :param s: match or delete
    :param sequence: the original sequence 
    :param path: the alignment path
    :returns: `.` for match and `_` for gaps
    """
    symbol = None
    if s == MATCH:
        return MATCH_CHAR
    else:
        return GAP_CHAR


def result_symbol_1d(i, s, sequence, path):
    """
    Build an alignment symbol

    :param i: position in path
    :param s: match or delete
    :param sequence: the original sequence 
    :param path: the alignment path
    :returns: `sequence[i-1]` for match and `_` for gaps
    """
    if s == MATCH:
        return str(int(sequence[path[i] - 1][0]))
    else:
        return GAP_CHAR


def result_symbol_nd(i, s, sequence, path):
    """
    Build an alignment symbol

    :param i: position in path
    :param s: match or delete
    :param sequence: the original sequence 
    :param path: the alignment path
    :returns: `sequence[i-1]` for match and `zeros` for gaps
    """
    if s == MATCH:
        return sequence[path[i] - 1]
    else:
        dim = sequence.shape[1]
        return np.zeros(dim)


def alignment_string(path, symbol_func, sequence = None):
    """
    Process a whole path

    :param sequence: the original sequence 
    :param path: the alignment path
    :param symbol_func: returns a symbol given the index, symbol, sequence and path
    :returns: list(symbols)
    """
    result = []
    for i, s in path_process(path):
        result.append(symbol_func(i, s, sequence, path))
    return result


class DolphinSequence(namedtuple("DolphinSequence", "file starts stops types embeddings")):

    @classmethod
    def empty(cls, filename):
        return cls(filename, [], [], [], [], [])

    def append(self, start, stop, signal_type, sil, embedding):
        self.starts.append(start)
        self.stops.append(stop)
        self.types.append(signal_type)
        self.silences.append(sil)
        self.embeddings.append(embedding)

    def align(self, other):
        x = self.embeddings
        y = other.embeddings
        print("Align sequences of length: {} {}".format(len(x), len(y)))
        dtw = DTW(max(len(x), len(y)))
        score, path = dtw.align(x, y)
        nd_x = np.stack(alignment_string([i for i, _ in path], result_symbol_nd, x))
        nd_y = np.stack(alignment_string([j for _, j in path], result_symbol_nd, y))
        return score, nd_x, nd_y


class AlignmentTests(unittest.TestCase):

    def test_align_1d(self):
        x = np.stack([np.ones(1) * i for i in range(0, 10)])
        y = np.stack([np.ones(1) * i for i in range(0, 10) if i % 2 == 0])
        dtw = DTW(max(len(x), len(y)))
        score, path = dtw.align(x, y)
        self.assertEqual(int(score), 5)
        self.assertEqual("".join(alignment_string([i for i, _ in path], result_symbol_1d, x)), '0123456789')
        self.assertEqual("".join(alignment_string([j for _, j in path], result_symbol_1d, y)), '0_2_4_6_8_')

    def test_align_str(self):
        x = np.stack([np.ones(1) * i for i in range(0, 10)])
        y = np.stack([np.ones(1) * i for i in range(0, 10) if i % 2 == 0])
        dtw = DTW(max(len(x), len(y)))
        score, path = dtw.align(x, y)
        self.assertEqual(int(score), 5)
        self.assertEqual("".join(alignment_string([i for i, _ in path], result_symbol_str)), '..........')
        self.assertEqual("".join(alignment_string([j for _, j in path], result_symbol_str)), '._._._._._')

    def test_align_nd(self):
        x = np.stack([np.ones(2) * i for i in range(0, 10)])
        y = np.stack([np.ones(2) * i for i in range(0, 10) if i % 2 == 0])
        dtw = DTW(max(len(x), len(y)))        
        score, path = dtw.align(x, y)
        nd_x = np.stack(alignment_string([i for i, _ in path], result_symbol_nd, x))
        nd_y = np.stack(alignment_string([j for _, j in path], result_symbol_nd, y))
        str_x = "".join([str(x) for x in nd_x])
        str_y = "".join([str(x) for x in nd_y])
        self.assertEqual(int(score), 10)
        self.assertEqual(str_x, '[0. 0.][1. 1.][2. 2.][3. 3.][4. 4.][5. 5.][6. 6.][7. 7.][8. 8.][9. 9.]')
        self.assertEqual(str_y, '[0. 0.][0. 0.][2. 2.][0. 0.][4. 4.][0. 0.][6. 6.][0. 0.][8. 8.][0. 0.]')