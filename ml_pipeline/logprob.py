from math import log, isinf, exp


ZERO = float('-inf')


class LogProb:

    def __init__(self, prob):
        self.prob = prob

    @classmethod
    def from_float(cls, prob):
        if prob == 0.0:
            return cls(ZERO)
        return cls(log(prob))

    @property
    def is_zero(self):
        return isinf(self.prob)

    @property
    def exp(self):
        if self.is_zero:
            return 0.0
        return exp(self.prob)

    def __mul__(self, other):
        assert isinstance(other, LogProb)   
        if self.is_zero or other.is_zero:
            return LogProb(ZERO)
        logprob = self.prob + other.prob
        return LogProb(logprob)

    def __truediv__(self, other):
        assert isinstance(other, LogProb) 
        if self.is_zero or other.is_zero:
            return LogProb(ZERO)
        logprob = self.prob - other.prob
        return LogProb(logprob)

    def __add__(self, other):
        assert isinstance(other, LogProb)        
        if self.is_zero:
            return LogProb(other.prob)
        if other.is_zero:
            return LogProb(self.prob)
        if self.prob > other.prob:
            return LogProb(self.prob + log(1.0 + exp(other.prob - self.prob)))
        return LogProb(other.prob + log(1.0 + exp(self.prob - other.prob)))

    def __gt__(self, other):
        return self.prob > other.prob

    def __repr__(self):
        return "p = {} [{}]".format(exp(self.prob), self.prob)