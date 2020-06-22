import unittest
import numpy as np

from ml_pipeline.health_checks import hamming_distance


class HammingDistance(unittest.TestCase):

    def test_hamming(self):
        x = np.zeros(10, dtype = np.int32)
        y = np.ones(10, dtype = np.int32)
        self.assertAlmostEqual(hamming_distance(x,x), 0.0, delta=1e-8)
        self.assertAlmostEqual(hamming_distance(x,y), 10.0, delta=1e-8)
        for i in range(0, 10):
            y[i] = 0
            self.assertAlmostEqual(hamming_distance(x,y), 10.0 - i - 1, delta=1e-8)
