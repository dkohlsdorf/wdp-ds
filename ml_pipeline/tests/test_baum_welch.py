import unittest
import numpy as np

import matplotlib.pyplot as plt

import ml_pipeline.fwd_bwd    as infer
import ml_pipeline.baum_welch as bw
from ml_pipeline.hidden_markov_model import HiddenMarkovModel
from ml_pipeline.tests.left_right_hmm import HMM_CONT
from ml_pipeline.markov_chain import Transition, START_STATE, STOP_STATE, DenseMarkovChain
from ml_pipeline.logprob import ZERO, LogProb


class BaumWelchTests(unittest.TestCase):

    LEFT_RIGHT = [
        [0.7, 0.3, 0.0],
        [0.0, 0.7, 0.3],
        [0.0, 0.0, 1.0]
    ]

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

    def test_markov(self):
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
        gamma, alpha, beta = infer.infer(HMM_CONT, seq) 
        gammas = [gamma]
        zetas  = [bw.infer(HMM_CONT, seq, alpha, beta)]    
        transitions = bw.markov(zetas, gammas)
        for i in range(0, 3):
            for j in range(0, 3):
                t = Transition(i, j)
                estimated = np.round(np.exp(transitions[t]), 1)
                expected  = np.round(BaumWelchTests.LEFT_RIGHT[i][j], 1)
                self.assertAlmostEqual(estimated, expected, delta=1e-12)
    
    def test_continuous_obs(self):
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
        hmm = HMM_CONT
        for _ in range(0, 10):
            gamma, _, _ = infer.infer(HMM_CONT, seq)
            gammas    = [gamma]
            sequences = [seq]
            obs       = bw.continuous_obs(sequences, gammas, min_variance=1)
            hmm.observations = obs
        self.assertEqual(round(obs[0].mean[0]), 0)
        self.assertEqual(round(obs[1].mean[0]), 1)
        self.assertEqual(round(obs[2].mean[0]), 0)
