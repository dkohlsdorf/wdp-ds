import tensorflow as tf
import numpy as np
from random import shuffle
from sklearn.metrics import accuracy_score


def merge_encoder(n_in, n_classes):
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
    
    def __init__(self, i, embedding, label, score, l = None, r = None):
        self.i         = i
        self.score     = score
        self.embedding = embedding
        self.left      = l
        self.right     = r
        self.label     = label
        
    def print(self, offset=""):
        print("{} {} {} {}".format(offset, self.i, self.score, np.mean(self.embeding)))
        if self.left is not None and self.right is not None:
            self.left.print(offset + "\t")
            self.right.print(offset + "\t")

    def merge(self, other, merger, training = True):
        merged         = merger([self.embedding, other.embedding])
        h              = merged[0]
        c              = merged[1]
        y              = merged[2]
        classification = merged[3]
        score = tf.nn.softmax_cross_entropy_with_logits(self.label, classification) \
            + tf.nn.softmax_cross_entropy_with_logits(y, c) \
            + self.score  \
            + other.score 
        if training:
          return Node(-1, h, self.label, score, self, other)
        else:
          return Node(-1, h, classification, score, self, other)


def ts2leafs(s, label, tokens, n_classes):
    sequence = []
    for i, x in enumerate(s):
        y = np.zeros((1, n_classes))
        y[0, label] = 1 
        node = Node(i, tokens[x], y, tf.constant(0.0))
        sequence.append(node)
    return sequence


def merge(x, m):
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
                    _y.append(node.label)
                    y.append(labels[i])
                    acc = accuracy_score(y, _y)
                    print("Score: {} {}/{} Acc:{} Epoch: {}".format(node.score[0], i, len(sequences), epoch))
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