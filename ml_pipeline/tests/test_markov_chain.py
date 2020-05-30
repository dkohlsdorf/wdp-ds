import unittest

from ml_pipeline.markov_chain import Transition, DenseMarkovChain
from ml_pipeline.logprob import *

class DenseMarkovChainTest(unittest.TestCase):
    LEFT_RIGHT = [
        [0.6, 0.4, 0.0],
        [0.0, 0.6, 0.4],
        [0.0, 0.0, 0.6]
    ]

    def test_from_probs(self):
        chain = DenseMarkovChain.from_probs(DenseMarkovChainTest.LEFT_RIGHT)
        for i in range(0, 3):
            for j in range(0, 3):
                trans = Transition(i, j)
                if i == j:
                    self.assertAlmostEqual(chain[trans].prob, LogProb.from_float(0.6).prob, delta=1e-8)
                elif j == i + 1:
                    self.assertAlmostEqual(chain[trans].prob, LogProb.from_float(0.4).prob, delta=1e-8)
                else:            
                    self.assertAlmostEqual(chain[trans].prob, ZERO, delta=1e-8)

    def test_getitem(self):
        chain = DenseMarkovChain.from_probs(DenseMarkovChainTest.LEFT_RIGHT)
        self.assertAlmostEqual(chain[Transition(0,0)].prob, LogProb.from_float(0.6).prob, delta=1e-8)
        self.assertAlmostEqual(chain[Transition(0,1)].prob, LogProb.from_float(0.4).prob, delta=1e-8)
        self.assertAlmostEqual(chain[Transition(0,2)].prob, ZERO, delta=1e-8)

    def test_setitem(self):
        chain = DenseMarkovChain(2)
        chain[Transition(0,0)]  = LogProb.from_float(1.0)
        chain[Transition(1,0)] += LogProb.from_float(2.0)
        chain[Transition(1,0)] += LogProb.from_float(3.0)
        self.assertAlmostEqual(chain[Transition(0,0)].prob, LogProb.from_float(1.0).prob, delta=1e-8)
        self.assertAlmostEqual(chain[Transition(1,0)].prob, LogProb.from_float(5.0).prob, delta=1e-8)

    def test_n_states(self):
        chain = DenseMarkovChain.from_probs(DenseMarkovChainTest.LEFT_RIGHT)
        self.assertEqual(chain.n_states, 3)