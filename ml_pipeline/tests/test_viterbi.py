import unittest
import numpy as np

from ml_pipeline.viterbi import viterbi
from ml_pipeline.logprob import LogProb

from ml_pipeline.tests.left_right_hmm import HMM_CONT

class ViterbiTest(unittest.TestCase):

    def test_viterbi(self):
        seq = np.array([
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1)
        ])
        path, ll = viterbi(HMM_CONT, seq)
        self.assertListEqual(list(path), [0,0,0,1,1,1,2,2,2])
        self.assertAlmostEqual(-1.5018560360449937, ll.prob, places=2)        

