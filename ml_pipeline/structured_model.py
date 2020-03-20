import tensorflow as tf

from tensorflow.keras.optimizers import *
from tensorflow.keras.layers     import *
from enum import Enum

from recursive import RecursiveNeuralNet

def token_dict(sequences):
    '''
    Builds a dictionary of tokens and labels

    :param sequences: list of dataframes with a field called 'cluster' for the tokens and 'behavior' for the label
    :returns: (cluster -> id), (id -> cluster), (behavior -> id), (id -> behavior)
    '''
    tokens = []
    labels = []
    for sequence in sequences:
        for label in sequence['behavior'][0]:
            labels.append(label)
        for c in sequence['cluster']:
            tokens.append(c)
    tokens = list(set(tokens))
    labels = list(set(labels))
    xindex = {}
    xreverse_index = {}
    i     = 1
    for token in tokens:
        xindex[token] = i
        xreverse_index[i] = token
        i += 1
    i = 0
    yindex = {}
    yreverse_index = {}
    for label in labels:
        yindex[label] = i
        yreverse_index[i] = label
        i += 1

    return xindex, xreverse_index, yindex, yreverse_index


def tokenize(sequence, token_dict):
    '''
    convert a sequence to the token ids

    :param sequence: a sequence of tokens
    :param token_dict: maps tokens to ids
    :returns: sequence -> ids
    '''
    vector = []
    for c in sequence:
        vector.append(token_dict[c])
    return vector


def sliding_window(sequence, win = 20, step = 4):
    '''
    A sliding window

    :param sequence: a sequence of tokens
    :param win: window size in number of tokens
    :param step: step size in number of tokens
    '''
    for i in range(min(len(sequence) - 1, win), len(sequence), step):
        window = sequence[i - win:i]
        yield window


class StructuredModels(Enum):
    CBOW = 1 # No Structure
    CONV = 2 # Subsequences in Communication
    LSTM = 3 # Sequential Structure
    RECU = 4 # Recursive  Structure


def annotation_model(xindex, yindex, max_len, model_type = StructuredModels.CBOW):
    '''
    A neural network that classifys discrete sequences

    :param xindex: token dict
    :param yindex: label dict
    :param max_len: maximum length of sequencesß
    :param model_type: the model_type
    :returns: model 
    '''
    n_in  = len(xindex)
    n_out = len(yindex)
    if model_type == StructuredModels.LSTM:
        model = tf.keras.Sequential([
            Embedding(n_in + 1, 32, input_length=max_len, mask_zero=True),
            BatchNormalization(),
            Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer='l2')),
            Bidirectional(LSTM(32, kernel_regularizer='l2')),
            Dropout(0.8),
            Dense(n_out, activation='softmax',  kernel_regularizer='l2')
        ])
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.summary()
    elif model_type == StructuredModels.RECU:
        model = RecursiveNeuralNet(n_out)
        model.compile(RMSprop())
    elif model_type == StructuredModels.CBOW:
        model = tf.keras.Sequential([
            Embedding(n_in + 1, 128, input_length=max_len, mask_zero=True),
            BatchNormalization(),
            Lambda(lambda x: tf.keras.backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2])),
            Dropout(0.8),
            Dense(n_out, activation='softmax', kernel_regularizer='l2')
        ])
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.summary()
    elif model_type == StructuredModels.CONV:
        model = tf.keras.Sequential([
            Embedding(n_in + 1, 32, input_length=max_len, mask_zero=True),
            BatchNormalization(),
            Conv1D(32, 3, kernel_regularizer='l2'),
            MaxPooling1D(2),
            Conv1D(32, 3, kernel_regularizer='l2'),
            MaxPooling1D(2),
            Conv1D(32, 3, kernel_regularizer='l2'),
            MaxPooling1D(2),
            Conv1D(32, 3, kernel_regularizer='l2'),
            MaxPooling1D(2),
            Flatten(),
            Dropout(0.8),
            Dense(n_out, activation='softmax', kernel_regularizer='l2')
        ])
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.summary()
    return model

