import unittest

import unittest
import numpy as np

from ml_pipeline.logprob import LogProb, ZERO
from ml_pipeline.fwd_bwd import fwd, bwd, infer

from ml_pipeline.tests.left_right_hmm import HMM_CONT

class FwdBwdTest(unittest.TestCase):

    @classmethod
    def dp2path(cls, dp):
        (T, N) = dp.shape
        path = []
        for t in range(0, T):
            max_ll    = ZERO
            max_state = 0
            for j in range(0, N):
                if dp[t, j] > max_ll:
                    max_ll = dp[t, j]
                    max_state = j
            path.append(max_state)
        return path

    def test_fwd(self):
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
        dp = fwd(HMM_CONT, seq)
        self.assertListEqual(FwdBwdTest.dp2path(dp), [0, 0, 1, 1, 1, 1, 2, 2, 2])

    def test_bwd(self):
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
        dp = bwd(HMM_CONT, seq)
        self.assertListEqual(FwdBwdTest.dp2path(dp), [0, 0, 0, 0, 1, 1, 1, 2, 2])

    def test_infer(self):
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
        dp, _, _ = infer(HMM_CONT, seq)
        self.assertListEqual(FwdBwdTest.dp2path(dp), [0, 0, 0, 1, 1, 1, 2, 2, 2])
