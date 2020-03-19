from enum import Enum
import tensorflow as tf


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
    CBOW = 1
    LSTM = 2
    CONV = 3
    RECU = 4

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
            tf.keras.layers.Embedding(n_in + 1, 128, input_length=max_len, mask_zero=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_out, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.summary()
    elif model_type == StructuredModels.CBOW:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(n_in + 1, 128, input_length=max_len, mask_zero=True),
            tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2])),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_out, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.summary()
    return model

