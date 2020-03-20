import tensorflow as tf
import numpy as np
from random import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix


def merge_encoder(n_in, n_classes):
    '''
    A semi supervised recursive neural network.
    Given inputs a, b the network will construct 
    a hidden layer and then reconstruct a and b
    as well as predict the classes.

    :params n_in: input dimension
    :params n_classes: number of classes
    :returns: the model
    '''
    a       = tf.keras.layers.Input(n_in)
    b       = tf.keras.layers.Input(n_in)
    c       = tf.keras.layers.Concatenate()([a,b])
    h       = tf.keras.layers.Dense(n_in,      activation = 'relu')(c)
    o       = tf.keras.layers.Dense(n_in * 2,  activation = 'softmax')(h)
    classes = tf.keras.layers.Dense(n_classes, activation = 'softmax')(h)
    merge   = tf.keras.models.Model(inputs=[a, b], outputs=[h, c, o, classes])
    merge.summary()
    return merge


class Node:
    '''
    A merge node with left and right children.
    Holds other information such the embedding of the merge,
    the ground truth label, the prediction of the network and
    the score.
    '''
    
    def __init__(self, i, embedding, label, prediction, score, l = None, r = None):
        self.i          = i
        self.score      = score
        self.embedding  = embedding
        self.left       = l
        self.right      = r
        self.label      = label
        self.prediction = prediction


    def merge(self, other, merger):
        '''
        Merge this node and another using a merge neural network
        score the merge as: score(reconstruction) + score(classififcation) + left_score + right_score
        '''
        merged         = merger([self.embedding, other.embedding])
        h              = merged[0]
        c              = merged[1]
        y              = merged[2]
        classification = merged[3]
        score = tf.nn.softmax_cross_entropy_with_logits(self.label, classification) \
            + tf.nn.softmax_cross_entropy_with_logits(y, c) \
            + self.score  \
            + other.score 
        return Node(-1, h, self.label, classification, score, self, other)


def ts2leafs(s, label, tokens, n_classes):
    '''
    Convert a labeled sequence to a sequence of leaf nodes

    :param s: sequence
    :param label: label for sequence
    :param tokens: token for each symbol in the sequence
    :param n_classes: number of classes
    :returns: sequence of leaf nodes
    '''
    sequence = []
    for i, x in enumerate(s):
        y = np.zeros((1, n_classes))
        y[0, label] = 1 
        node = Node(i, tokens[x], y, None, tf.constant(0.0))
        sequence.append(node)
    return sequence


def merge(x, m):
    '''
    merge a sequence to one node using the merge neural network

    :param x: sequence
    :param m: merge model
    :returns: one node
    '''
    while len(x) > 1:        
        min_loss = float('inf')
        min_node = None
        min_i = 0
        min_j = 0
        for i in range(len(x)):
            for j in range(len(x)):
                if i < j:
                    node = x[i].merge(x[j], m)
                    if node.score < min_loss:
                        min_node = node
                        min_loss = node.score
                        min_i = i
                        min_j = j
        #print("Merge: {} {}".format(min_i, min_j))
        x[min_i] = min_node
        x = [x[idx] for idx in range(0, len(x)) if idx != min_j]
    return x[0]


class RecursiveNeuralNet:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        
    def compile(self, optimizer):
        self.optimizer = optimizer

    def fit(self, sequences, labels, epochs=5):
        clusters = [c for c in sequences.flatten()]
        tokens   = dict([(c, i) for i, c in enumerate(sorted(list(set(clusters))))])
        bits     = int(np.ceil(np.log(len(tokens)) / np.log(2)))
        self.merger = merge_encoder(bits, self.n_classes)

        for c, i in tokens.items():
            tokens[c] = np.float32([int(c) for c in np.binary_repr(i, width = bits)]).reshape(1, bits)
        for epoch in range(epochs):
            _y = []
            y  = []
            for i, sequence in enumerate(sequences):
                x = ts2leafs(sequence, labels[i], tokens, self.n_classes)
                with tf.GradientTape(watch_accessed_variables=True) as tape:
                    tape.watch(self.merger.variables) 
                    node = merge(x, self.merger)
                    _y.append(np.argmax(node.prediction))
                    y.append(labels[i])
                    acc = accuracy_score(y, _y)
                    print("Score: {} {}/{} Epoch: {}".format(node.score[0], i, len(sequences), epoch))
                    if len(y) == 10:
                        print("Acc: {}".format(acc))
                        print(confusion_matrix(y, _y))
                        _y = []
                        y  = []

                    g    = tape.gradient(node.score, self.merger.variables)
                    self.optimizer.apply_gradients(zip(g, self.merger.variables))

    def predict_structured(self, sequences):
        predictions = [] 
        for i, sequence in range(len(sequences)):
            x    = ts2leafs(sequence, None)
            node = merge(x, self.merger)
            predictions.append(node)
        return node

    def predict(self, sequences):
        return [node.label for node in self.predict_structured(sequences)]