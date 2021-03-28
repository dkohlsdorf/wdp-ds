import pandas as pd
import os

from collections import namedtuple


Symbol   = namedtuple('Symbol', 'id type')
Sequence = namedtuple('Sequence', 'symbols file offset')


def extract_id(filename):
    return filename.split('_')[0]


def extract_offset(filename):
    return filename.split('_')[1].replace('.csv', '')


def extract_sequences(files):
    sequences = []
    for file, path in files:
        shotid        = extract_id(file)
        offset        = extract_offset(file)                
        df            = pd.read_csv(path)
        symbols = []
        for i, row in df.iterrows():
            s = Symbol(row['cluster'], row['labels'])
            symbols.append(s)
        sequence = Sequence(symbols, shotid, offset)
        sequences.append(sequence)
    return sequences
