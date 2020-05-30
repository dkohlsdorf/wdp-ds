import unittest
import numpy as np

from ml_pipeline.dtw import DTW


class DTWTest(unittest.TestCase):

    X = np.stack([np.zeros(1), np.ones(1), np.zeros(1), np.ones(1)])
    Y = np.stack([np.zeros(1), np.ones(1), np.ones(1), np.zeros(1)])
    ALIGN = DTW(4)

    def test_dtw_equal(self):
        distance, path = DTWTest.ALIGN.align(DTWTest.X, DTWTest.X) 
        self.assertAlmostEqual(distance, 0.0, delta=1e-8)
        self.assertListEqual(list(path), [(1, 1), (2, 2), (3, 3), (4, 4)])

    def test_dtw_different(self):
        distance, path = DTWTest.ALIGN.align(DTWTest.X, DTWTest.Y) 
        self.assertAlmostEqual(distance, 1.0, delta=1e-8)
        self.assertListEqual(list(path), [(1, 1), (2, 2), (2, 3), (3, 4), (4, 4)])
