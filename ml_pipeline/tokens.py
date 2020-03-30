
import numpy as np
import matplotlib.pyplot as plt


class Tokens:
    '''
    A token dictionary mapping tokens to ids
    and ids to tokens.
    '''
    def __init__(self):
        self.fwd = {}
        self.bwd = {}
        self.cur = 0

    def add(self, i):
        '''
        Insert a token 

        :param i: a token
        '''
        if i not in self.fwd:
            self.fwd[i]   = self.cur
            self.bwd[self.cur] = i
            self.cur += 1

    def max_id(self):
        '''
        Maximum id in the dictionary

        :returns: max id assigned to a token
        '''
        return max(self.fwd.values()) 

    def convert(self, seq, fwd=True):
        '''
        Convert a sequence of tokens to ids or the
        other way around

        :param seq: sequence of tokens or ids
        :param fwd: if true we convert tokens to ids 
        :returns: sequence of ids or tokens 
        '''
        if fwd:
            return [self.fwd[i] for i in seq]
        else:
            return [self.bwd[i] for i in seq]


class CountTokens:
    '''
    Count tokens in each document.
    '''
    def __init__(self, n_in, n_out):
        self.fwd = np.zeros((n_out, n_in))
        self.bwd = np.zeros((n_in, n_out))
        self.closed = set([])

    def add(self, token, document):
        '''
        Add a token in a document

        :param token: a token
        :param document: a document id
        '''
        self.fwd[document][token] += 1
        if (token, document) not in self.closed:
            self.bwd[token][document] += 1
            self.closed.add((token, document))
            
    def tfidf(self):
        '''
        Tf-IDF for each token and document
        '''
        n, m  = self.fwd.shape
        tfidf = np.zeros((n, m))
        for document in range(n):
            n_tokens = np.sum(self.fwd[document])
            for i in range(0, m):
                idf = np.log(1 + np.sum(self.bwd[i]) / n)
                tf  = np.log(1 + self.fwd[document][i] / (10 * n_tokens))
                tfidf[document][i] = tf * idf
        return tfidf


def iter_ngram(sequence, n):
    '''
    Generate ngrams tokens from a sequence

    :param sequence: a sequence
    :param n: as in ngram
    '''
    for i in range(n, len(sequence)):
        yield "_".join([str(x) for x in sequence[i-n:i]])


def plot_idf(sequence, labels, output_folder, counts = False, top_k = 10, width=50):
    '''
    Plot the top tf-idf vectors for each label

    :param sequence: a sequence of tokens
    :param labels: a sequence of labels
    :param output_folder: where to save
    :param counts: output counts or idf
    :param top_k: top k entries
    :param width: plot width
    '''
    input_tokens = Tokens()
    label_tokens = Tokens()
    for i in sequence:
        input_tokens.add(i)
    for i in labels:
        label_tokens.add(i)

    pdf = CountTokens(input_tokens.max_id() + 1, label_tokens.max_id() + 1)
    s   = input_tokens.convert(sequence)
    l   = label_tokens.convert(labels)

    for i in range(len(s)):
        pdf.add(s[i], l[i])
    if counts:
        tfidf = pdf.fwd
    else:
        tfidf = pdf.tfidf()
    n, m  = tfidf.shape

    for i in range(n):
        y = []
        for j in range(0, m):
            y.append((input_tokens.bwd[j], tfidf[i][j]))
        y = sorted(y, key = lambda x: -x[1])[2:top_k]

        x     = np.arange(top_k - 2)
        score = [x[1] for x in y]
        names = [x[0] for x in y] 
        
        plt.figure(figsize=(width, 50))
        plt.bar(x, score)
        plt.xticks(x, names, rotation=90)
        plt.title(label_tokens.bwd[i])
        plt.savefig('{}/{}'.format(output_folder, label_tokens.bwd[i]))
        plt.close()
