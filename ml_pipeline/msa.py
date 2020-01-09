import numpy as np
import random
from enum import Enum


class Mutations(Enum):
    INSERT_GAP  = 0
    DELETE_GAP  = 1
    SWAP_ROWS   = 2

    
class CrossOver(Enum):
    HORIZONTAL = 0
    VERTICAL   = 1

    
class GeneticMSA:

    GAP = -1
    
    def __init__(self, data, msa, borders):
        self.data    = data    
        self.msa     = msa
        self.borders = borders
        
    @property
    def n_sequences(self):
        return self.msa.shape[0]
    @property
    def max_length(self):
        return self.msa.shape[1]

    def gaps(self):
        for i in range(0, self.n_sequences):
            for j in range(0, self.borders[i]):
                if self.msa[i][j] == GAP:
                    yield (i, j)

    def breed_with(self, other):
        cmd = random.randint(0, 1)
        if cmd == CrossOver.HORIZONTAL:
            next = self.borders.copy()
            next_borders = self.borders.copy()
            for i in range(0, self.n_sequences):
                if random.uniform(0, 1) > 0.5:
                    next[i, :] = other.msa[i, :]
                    next_borders[i] = other.borders[i]
            return Alignment(self.data, next, next_borders)
        elif cmd == CrossOver.VERTICAL:
            pass
        
    def mutate(self):
        cmd = random.randint(0, 3)
        if cmd == Mutations.INSERT_GAP:
            next = self.msa.copy()
            i = random.randint(0, self.n_sequences)
            t = random.randint(0, self.borders[i])
            tmp = next[i, t:self.border[i]].copy()
            next[i, t] = GAP
            next[i, t + 1:self.border[i] + 1] = tmp
            next_borders = self.borders.copy()
            next_borders[i] + 1
            return Alignment(self.data, next, next_borders)
        elif cmd == Mutations.DELETE_GAP:
            gaps = self.gaps([gap for gap in self.gaps()])
            n = len(gaps)
            gap_id = random.randint(0, n)
            (i, t) = gaps[gap_id]
            next = self.msa.copy()
            next[i, t:self.border[i] - 1] = tmp[i, t + 1:self.border[i]]
            next[i, self.border[i]] = GAP
            next_borders = self.borders.copy()
            next_borders[i] - 1
            return Alignment(self.data, next, next_borders)
        elif cmd == Mutations.SWAP_ROWS:
            i = random.randint(0, self.n_sequences)
            j = random.randint(0, self.n_sequences)
            while i == j:
                j = random.randint(0, self.n_sequences)
            next = self.msa.copy()
            tmp_row = next[i, :].copy()
            next[i, :] = next[j, :]
            next[j, :] = tmp_row
            next_borders = self.border
            tmp_border = next_borders[i]
            next_borders[i] = next_borders[j]
            next_borders[j] = tmp_border
            return Alignment(self.data, next, next_borders)               

def initial_population(data, population_size):
    pass

def select(population, n_offspring):
    pass

def mutate(mutants):
    pass

def crossover(selection):
    pass

def kill(population, population_size):
    pass

def best(population):
    pass

def evolve(data, n_epochs, population_size, n_offspring):
    population = initial_population(data, population_size)
    for i in range(0, n_epochs):
        selection = select(population, n_offspring)
        mutants = mutate(crossover(selection))
        population.extend(mutants)
        population = kill(population, population_size)
    return best(population) 
