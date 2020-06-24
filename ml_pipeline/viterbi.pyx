# Methods for Inference
#
# REFERENCES:
# [RAB] Rabiner: "A tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", Proceedins of the IEEE, 1989
# [HOL] John and Wendy Holmes: "Speech Synthesis and Recognition", Taylor & Francis Ltd; Second Edition, 2001


import numpy as np
from markov_chain import START_STATE, STOP_STATE, Transition
from logprob import ZERO, LogProb


def viterbi(hmm, sequence): 
    """
    Align a sequence to a hidden Markov model.

    Implements [RAB] equation 31 and [HOL] equation 9.11

    Suggested efficiency improvement: Do not save the whole matrix for dynamic
    programming but only the last time slice

    :param hmm: a hidden Markov model
    :param sequence: a sequence of length M and dimension d
    :returns: path and alignment score
    """
    cdef int T = len(sequence)
    cdef int N = hmm.n_states
    
    cdef double[:, :] dp  = np.ones((T, N), dtype=np.double) * ZERO # dynamic programming matrix
    cdef long[:, :] bp    = np.zeros((T, N), dtype=np.int)          # back tracking matrix
    cdef long[:]  path    = np.zeros(T, dtype=np.int)
    
    cdef double ll  = 0.0
    cdef int argmax = 0

    cdef int t, i, j
    for i in range(0, N):
        init     = Transition(START_STATE, i)
        sample   = sequence[0]
        logprob  = hmm.observations[i][sample] * hmm.transitions[init] 
        dp[0, i] = logprob.prob

    for t in range(1, T):
        for i in range(0, N):
            max_ll = LogProb.from_float(0.0)
            for j in range(0, N):
                transition = Transition(j, i)       
                logprob = LogProb(dp[t - 1, j]) * hmm.transitions[transition] 
                if logprob > max_ll:
                    max_ll = logprob
                    argmax = j
            sample   = sequence[t]     
            logprob  = max_ll * hmm.observations[i][sample]
            dp[t, i] = logprob.prob
            bp[t, i] = argmax

    max_ll = LogProb.from_float(0.0)
    argmax = 0
    for i in range(0, N):
        end = Transition(i, STOP_STATE)
        logprob = hmm.transitions[end] * LogProb(dp[T - 1, i])
        dp[T - 1, i] = logprob.prob 
        if LogProb(dp[T - 1, i]) > max_ll:
            max_ll = logprob
            argmax = j

    path[T - 1] = argmax
    t = T - 2
    i = bp[T - 1, argmax]
    while t >= 0:
        path[t] = i
        i = bp[t, i]
        t -= 1
    return np.asarray(path), LogProb(max_ll.prob) 