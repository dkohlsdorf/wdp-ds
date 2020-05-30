import numpy as np
import io
from sklearn.mixture import GaussianMixture
from ml_pipeline.logprob import LogProb, ZERO


class Gaussian:

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def __getitem__(self, x):
        '''
        Allows the use of the distribution as pdf[x] = Normal(x|mu,sigma)

        :param x: a nd-sample
        '''
        error = -0.5 * (np.square(x - self.mean) / self.variance)
        return LogProb(self.scaler + np.sum(error))

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def dim(self):
        return len(self.variance)

    @property
    def scaler(self):
        return -((self.dim / 2) * np.log(2 * np.pi)) - np.sum(np.log(np.sqrt(self.variance)))