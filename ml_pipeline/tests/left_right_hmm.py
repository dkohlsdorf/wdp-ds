import numpy as np

from ml_pipeline.hidden_markov_model import HiddenMarkovModel
from ml_pipeline.distributions import Gaussian
from ml_pipeline.markov_chain import DenseMarkovChain, Transition, START_STATE, STOP_STATE
from ml_pipeline.logprob import LogProb


LEFT_RIGHT = DenseMarkovChain.from_probs([
    [0.6, 0.4, 0.0],
    [0.0, 0.6, 0.4],
    [0.0, 0.0, 0.6]
])

LEFT_RIGHT[Transition(START_STATE, 0)] = LogProb(0.0)
LEFT_RIGHT[Transition(2, STOP_STATE)]  = LogProb(0.0)

CONT = [
    Gaussian(np.zeros(1), np.ones(10)),
    Gaussian(np.ones(1),  np.ones(10)),
    Gaussian(np.zeros(1), np.ones(10))
]

HMM_CONT = HiddenMarkovModel(LEFT_RIGHT, CONT)
