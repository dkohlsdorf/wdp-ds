import numpy as np

from logprob import ZERO, LogProb
from markov_chain import Transition
from distributions import Gaussian


def infer(hmm, sequence, fwd, bwd):
    cdef int T = len(sequence)
    cdef int N = hmm.n_states    
    cdef double[:, :, :] zeta = np.ones((T - 1, N, N), dtype=np.double) * ZERO
    cdef int t, i, j
    for t in range(0, T - 1):
        norm = LogProb(ZERO)
        for i in range(0, N):
            for j in range(0, N):
                trans    = Transition(i, j)
                sample   = sequence[t + 1]

                logprob  = LogProb(fwd[t][i]) * hmm.transitions[trans] 
                logprob *= hmm.observations[j][sample]
                logprob *= LogProb(bwd[t + 1][j])

                norm         += logprob
                zeta[t][i][j] = logprob.prob
                
        for i in range(0, N):
            for j in range(0, N):
                zeta[t][i][j] -= norm.prob
    return np.asarray(zeta)


def markov(zetas, gammas):
    assert len(zetas) > 0 and len(zetas) == len(gammas)
    cdef int m = len(zetas)
    cdef int n = zetas[0].shape[1]
    cdef double[:, :] transitions = np.ones((n, n), dtype=np.double) * ZERO
    cdef int e, t, i, j
    for i in range(0, n):
        for j in range(0, n):
            scaler = LogProb(ZERO)
            for e in range(0, m):
                T = zetas[e].shape[0]
                for t in range(0, T):
                    logprob = LogProb(zetas[e][t,i,j]) + LogProb(transitions[i, j])
                    print("LOGPROB: {}".format(logprob))
                    transitions[i,j] = logprob.prob
                    scaler += LogProb(gammas[e][t, i])
            print("SCALER: {}".format(scaler))
            prob = LogProb(transitions[i,j]) / scaler
            transitions[i, j] = prob.prob
    return np.asarray(transitions)


def continuous_obs(sequences, gammas, min_variance=1.0):
    assert len(gammas) > 0 and len(sequences) == len(gammas)
    cdef int m = len(gammas)
    cdef int n = gammas[0].shape[1]
    cdef int d = len(sequences[0][0])
    cdef observations = []
    cdef int i, e, t, j
    for i in range(0, n):
        mu     = np.zeros(d)
        sigma  = np.zeros(d) 
        scaler = 0 
        for e in range(0, m):
            T = gammas[e].shape[0]
            for t in range(0, T):
                weight = LogProb(gammas[e][t, i]).exp
                mu     += sequences[e][t] * weight
                scaler += weight
        mu /= scaler
        for e in range(0, m):
            T = gammas[e].shape[0]
            for t in range(0, T):
                weight = LogProb(gammas[e][t, i]).exp
                sigma += np.square(sequences[e][t] - mu) * weight
        sigma /= scaler
        for j in range(0, d):
            sigma[j] = max(min_variance, sigma[j])
        observations.append(Gaussian(mu, sigma))
    return observations