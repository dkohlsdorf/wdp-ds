import numpy as np
import copy
import itertools
import pandas as pd

import logging
logging.basicConfig()
log = logging.getLogger('gp')
log.setLevel(logging.INFO)

AND_SYMBOL    = -1
OR_SYMBOL     = -2
DONT_CARE     = -3
REPEAT        = -4


class RegexpNode:
    
    def __init__(self, symbol, children = []):
        self.symbol = symbol
        self.children = children
            
    @classmethod
    def from_string(cls, string):
        if len(string) == 1:
            return cls(string[0])
        else:
            return cls(AND_SYMBOL, [cls(string[0]), RegexpNode.from_string(string[1:])])
                    
    @property
    def is_leaf(self):
        return len(self.children) == 0 
    
    @property
    def is_binary(self):
        return len(self.children) == 2
    
    def cost(self, cost):
        if self.is_leaf:
            if self.symbol == DONT_CARE:
                return cost[DONT_CARE]
            else:
                return 0
        else:
            subtree_cost = cost[self.symbol]
            for c in self.children:
                subtree_cost += c.cost(cost)
            return subtree_cost
                
    def n_nodes(self):
        if self.is_leaf:
            return 1
        else:
            n = 0
            for child in self.children:
                n += child.n_nodes()
            return n
    
    def random_path(self):
        if self.is_leaf:
            return [self]
        next_child = np.random.randint(0, len(self.children))
        return [self] + self.children[next_child].random_path()
    
    def random_leaf(self):
        return self.random_path()[-1]
    
    def random_internal(self):
        p = [node for node in self.random_path()[0:-1]]
        i = np.random.randint(0, len(p))
        return p[i]

    def random_binary(self):
        p = [node for node in self.random_path()[0:-1] if node.is_binary]
        i = np.random.randint(0, len(p))
        return p[i]
    
    def __str__(self):
        if self.is_leaf:
            if self.symbol == DONT_CARE:
                return "."
            return str(self.symbol)
        elif self.symbol == AND_SYMBOL:
            return str(self.children[0]) + str(self.children[1])
        elif self.symbol == OR_SYMBOL:
            return "(" + str(self.children[0]) + "|" + str(self.children[1]) + ")"
        elif self.symbol == REPEAT:
            return "(" + str(self.children[0]) + ")" + "+"
        
    
def crossover(tree_a, tree_b):
    '''
    Select a random subtree in a and b and
    swap them
    
    :param tree_a: first parent to crossover
    :param tree_b: second parent to crossover
    :returns: two new trees
    '''
    tree_a = copy.deepcopy(tree_a)
    tree_b = copy.deepcopy(tree_b)
    node_a = tree_a.random_internal()
    node_b = tree_b.random_internal()
    child_a = np.random.randint(0, len(node_a.children))
    child_b = np.random.randint(0, len(node_b.children))
    
    tmp                      = node_a.children[child_a]
    node_a.children[child_a] = node_b.children[child_b]
    node_b.children[child_b] = tmp
    return tree_a, tree_b
    
    
def mutate_terminal(tree, symbol):
    '''
    Search a terminal and add a don't care symbol
    
    :param tree: the tree to mutate
    :param symbol: the symbol to replace with
    
    :returns: a mutated tree
    '''
    tree = copy.deepcopy(tree)
    node = tree.random_leaf()
    node.symbol = symbol    
    return tree
    
    
def mutate_insert(tree, insert_symbol, symbol=AND_SYMBOL, right=False):
    '''
    Search a terminal and replace it with an and node.
    The left terminal will hold the symbol and the right terminal the symbol of the random node
    
    :param tree: the tree to mutate
    :param insert_symbol: the symbol to insert
    :param symbol: operation symbol: AND / OR
    :param right: insert left or right
    
    :returns: a mutated tree
    '''
    tree = copy.deepcopy(tree)
    and_node   = tree.random_leaf()
    node_left  = RegexpNode(insert_symbol)
    node_right = RegexpNode(and_node.symbol)
    if right:
        node_right = RegexpNode(insert_symbol)
        node_left = RegexpNode(and_node.symbol)
    and_node.symbol   = symbol
    and_node.children = [node_left, node_right]
    return tree


def mutate_repeat(tree):
    '''
    Search a binary node and repeat one of it's children
    
    :param tree: the tree to mutate
    :returns: a mutated tree
    '''
    tree = copy.deepcopy(tree)
    node  = tree.random_binary()
    child = np.random.randint(0, len(node.children))
    tmp   = node.children[child]
    node.children[child] = RegexpNode(REPEAT, [tmp])
    return tree


def mutate_binary(tree):
    '''
    Flip an and to an or or the other way around
        
    :param tree: the tree to mutate
    :returns: a mutated tree
    '''
    tree = copy.deepcopy(tree)
    node = tree.random_binary()
    if node.symbol == AND_SYMBOL:
        node.symbol = OR_SYMBOL
    else:
        node.symbol = AND_SYMBOL
    return tree
    
    
def match(string, regexp, depth = 0):
    '''
    Match a string with a regular expression in tree form.
    
    :param string: an integer list
    :param regexp: regular expression tree
    :returns: true if the string matches the expression
    '''
    if regexp.symbol >= 0:
        return len(string) == 1 and regexp.symbol == string[0]
    elif regexp.symbol == DONT_CARE:
        return len(string) == 1
    elif regexp.symbol == AND_SYMBOL:
        assert len(regexp.children) == 2
        result = False
        for i in range(0, len(string) + 1):
            result = result or match(string[0:i], regexp.children[0], depth + 1) and match(string[i: len(string)], regexp.children[1], depth + 1)
        return result
    elif regexp.symbol == OR_SYMBOL:
        assert len(regexp.children) == 2
        result = match(string, regexp.children[0], depth + 1) or match(string, regexp.children[1], depth + 1)
        return result
    elif regexp.symbol == REPEAT:
        if len(string) > 0:
            matches = -1
            for i in range(0, len(string) + 1):
                if match(string[0: i], regexp.children[0], depth + 1):
                    matches = i
            if matches < 1:
                return False
            return match(string[matches: len(string)], regexp, depth + 1)
        return True
    
    
def score(expression, examples, labels, cost, n_labels = 2):
    '''
    Maximum match probability for each label including a match prior
    
    :param expression: regular expression tree
    :examples: a set of n example strings
    :labels: a set of n labels
    :param n_labels: number of labels
    :returns: score
    '''
    matches    = np.zeros(n_labels) 
    size_prior = np.exp(-0.01 * expression.n_nodes())
    spec_prior = np.exp(-0.01 * expression.cost(cost))

    for example, label in zip(examples, labels):
        if match(example, expression):
            matches[label] += 1
    matches /= sum(matches) + 1
    matches *= size_prior * spec_prior
    return max(matches), np.argmax(matches)


TERMINAL = 0
INSERT   = 1
LOOP     = 2
BINARY   = 3 


def random_mutation(expression, symbols, commands = [TERMINAL, INSERT, LOOP, BINARY]):
    '''
    Apply a random  mutation to the expression
    
    :param symbols: possible terminals in the grammar including don't care symbol
    :param commands: basically the non terminals

    :returns a new expression.
    '''
    cmd = np.random.randint(0, len(commands))
    if cmd == TERMINAL:
        symbol = np.random.randint(0, len(symbols))
        return mutate_terminal(expression, symbols[symbol])
    if cmd == INSERT:
        insert_symbol = np.random.randint(0, len(symbols))
        operation = AND_SYMBOL
        right     = np.random.rand() > 0.5
        if np.random.rand() > 0.5:
            operation = OR_SYMBOL
        return mutate_insert(expression, symbols[insert_symbol], operation, right)
    if cmd == LOOP:
        return mutate_repeat(expression)
    if cmd == BINARY:
        return mutate_binary(expression)
    

def dedup(expressions):
    '''
    Delete all duplicate expressions
    
    :param expressions: all expressions
    '''
    closed = set()
    deduplicated = []
    for e in expressions:
        if str(e) not in closed:
            closed.add(str(e))
            deduplicated.append(e)
    return deduplicated


def evolve(examples, labels, cost, n_labels = 2, symbols=[0, 1, DONT_CARE], epochs=25, n_candidates=128, pop_size=1024):
    '''
    Evolve regular expression to match labeled strings.
    
    :param examples: a list of integer lists 
    :param labels: the labels associated with each integer list
    :param cost: the cost of each operation to score each regular expression
    :param n_labels: the number of labels
    :param symbols: all possible symbols including don't care
    :param epochs: how many iterations do we use
    :param n_candidates: number of candidates selected for mutation / crossover
    :param pop_size: the total population size
    :returns: `pop_size` many solutions ranked by the score
    '''
    expressions   = [RegexpNode.from_string(example) for example in examples]
    scores_labels = [score(expression, examples, labels, cost, n_labels) for expression in expressions] 
    scores        = [s for s, _ in scores_labels]
    labeling      = [l for _, l in scores_labels]
                        
    log.info("Initial Pop: {}".format(max(scores)))
    for epoch in range(epochs):            
        # Score all expressions in the population and take the k best
        result = [(e, s, l) for e, s, l in zip(expressions, scores, labeling)]
        result.sort(key=lambda x: -x[1])
        candidates  = [e for e, _, _ in result[0:n_candidates]]
        
        # Apply mutations and crossover to the candidates
        mutants     = [random_mutation(c, symbols) for c in candidates]
        np.random.shuffle(candidates)
        children    = [crossover(candidates[i], candidates[i - 1]) for i in range(1, min(n_candidates, len(candidates)))]
        children    = list(itertools.chain.from_iterable(children))
    
        # Rescore the offspring together with the parents and select the top population, discard the rest
        expressions = dedup(expressions + mutants + children)    
        scores_labels = [score(expression, examples, labels, cost, n_labels) for expression in expressions] 
        scores        = [s for s, _ in scores_labels]
        labeling      = [l for _, l in scores_labels]
        result        = [(e, s, l) for e, s, l in zip(expressions, scores, labeling)]
        result.sort(key=lambda x: -x[1])
        expressions = [e for e, _, _ in result[0:pop_size]]
        scores      = [s for _, s, _ in result[0:pop_size]]
        labeling    = [l for _, _, l in result[0:pop_size]]
        log.info("Evolved Pop: {} {} {}".format(epoch, sum(scores), len(expressions)))
    return expressions, scores, labeling
