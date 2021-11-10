import pandas as pd
import itertools

from collections import Counter
from lib_dolphin.htk_helpers import *


class Strings(namedtuple('Strings', 'l1 l2 label')):
            
    def ngrams_l1(self, n):
        if len(self.l1) >= n:
            return [" ".join(self.l1[i-n:i]) for i in range(n, len(self.l1))]

    def ngrams_l2(self, n):
        if len(self.l2) >= n:
            return [" ".join(str(self.l2[i-n:i])) for i in range(n, len(self.l2))]

    
def parse_l2(sequencing_file, T, FFT_STEP, perc=0.01, do_compress=False):
    annotations = parse_mlf(sequencing_file)
    win         = T // 2 * FFT_STEP
    likelihoods = []
    labels      = []
    regions     = [] 
    for f, x in annotations.items():
        if len(x) > 1 and do_compress:
            x = compress(x)
        for start, stop, c, ll in x:
            start *= win
            stop  *= win
            likelihoods.append(ll)
            labels.append(c)
            regions.append((f, start, stop))
    likelihoods = sorted(likelihoods)
    th = int(len(likelihoods) * perc)
    starts = []
    stops  = []
    files  = []
    lab    = []
    for i in range(0, len(regions)):
        if likelihoods[i] < th and labels[i] != 'sil':
            file, start, stop = regions[i]
            lab.append(labels[i])
            starts.append(start)
            stops.append(stop)
            files.append(file)
    return starts, stops, files, lab
    
    
def parse_l1(sequencing_file, T, FFT_STEP):
    annotations = pd.read_csv(sequencing_file)
    win         = T // 2 * FFT_STEP
    annotations['start'] = annotations['start'].apply(lambda x: x * win)
    annotations['stop']  = annotations['stop'].apply(lambda x: x * win)
    annotations['filenames']  = annotations['filenames'].apply(lambda x: x.replace('.wav', ''))

    starts = list(annotations['start'])
    stops  = list(annotations['stop'])
    files  = list(annotations['filenames'])
    label  = list(annotations['label'])
    return starts, stops, files, label


def by_file(files, symbols):
    f = {}
    cur_f = files[0]
    cur_symb = [symbols[0]]
    for i in range(1, len(files)):
        if files[i] != cur_f:
            f[cur_f] = cur_symb
            cur_f = files[i]
            cur_symb = [symbols[i]]
        else:
            cur_symb.append(symbols[i])
    return f


def ngram_statistics(l1, l2, label_f, T, FFT_STEP, N = 10):
    l1_starts, l1_stops, l1_files, l1_labs = parse_l1(l1, T, FFT_STEP)
    l2_starts, l2_stops, l2_files, l2_labs = parse_l2(l2, T, FFT_STEP)
    l1_byfile = by_file(l1_files, l1_labs)
    l2_byfile = by_file(l2_files, l2_labs)
    stats = {}
    labels = set()
    for key in l1_byfile.keys():
        label = label_f(key)
        labels.add(label)
        l2 = [] if key not in l2_byfile else l2_byfile[key]
        if label not in stats:
            stats[label] = []
        stats[label].append(Strings(l1_byfile[key], l2, label))

    counters = { key: Counter([]) for key in labels}
    for i in range(1, 10):
        for l in labels:
            l1 = list(itertools.chain(*[strg.ngrams_l1(i) for strg in stats[l] if strg.ngrams_l1(i) is not None]))
            l2 = list(itertools.chain(*[strg.ngrams_l2(i) for strg in stats[l] if strg.ngrams_l2(i) is not None]))
            counters[l] += Counter(l1) + Counter(l2)
    print(counters)
    