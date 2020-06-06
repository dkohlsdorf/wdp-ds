import unittest
import numpy as np

from ml_pipeline.sequence_hashing import *


class SequenceHashingTest(unittest.TestCase):

    X = np.stack([np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.ones(1), np.ones(1), np.ones(1)])

    def test_paa(self):
        compressed = paa(SequenceHashingTest.X, 2)
        self.assertAlmostEqual(compressed[0][0], 0.0)
        self.assertAlmostEqual(compressed[1][0], 1.0)

    def test_sax(self):
        compressed = multidim_sax([SequenceHashingTest.X], 2, 2)
        self.assertEqual(len(set(compressed[0])), 2)
        self.assertTrue(0 in set(compressed[0]))        
        self.assertTrue(1 in set(compressed[0]))