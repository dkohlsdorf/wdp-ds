import os
import numpy as np
import struct
import pandas as pd
import sys
import glob
import pickle as pkl
import random

from lib_dolphin.eval import *
from subprocess import check_output


FLOOR = 1.0
PERIOD      = 1
SAMPLE_SIZE = 4 
USER        = 9
ENDIEN      = 'big'


def write_htk(sequence, to, norm = False):
    n = len(sequence)
    dim = len(sequence[0])
    with open(to, "wb") as f:        
        sze = dim * SAMPLE_SIZE
        f.write(n.to_bytes(4, byteorder=ENDIEN))
        f.write(PERIOD.to_bytes(4, byteorder=ENDIEN))
        f.write(sze.to_bytes(2, byteorder=ENDIEN))
        f.write(USER.to_bytes(2, byteorder=ENDIEN))
        for i in range(0, n):
            if norm:
                mu  = np.mean(sequence[i])
                std = np.std(sequence[i]) + 1.0
            for j in range(0, dim):
                x = sequence[i, j]
                if np.isnan(x):
                    print(to)
                if norm:
                    ba = bytearray(struct.pack(">f", (x - mu) / std))
                else:
                    ba = bytearray(struct.pack(">f", x))
                f.write(ba)

                
def vec(v):    
    line = " ".join([str(x) for x in v])
    return "\t{}".format(line)


def mat(x):
    return "\n".join([vec(x[i]) for i in range(len(x))])


def left_right_hmm(max_states, dims, name="proto"):
    transitions = np.zeros((max_states, max_states))    
    means       = np.zeros(dims)
    variances   = np.ones(dims)  

    transitions[0, 1] = 1.0
    states = []
    for i in range(1, max_states - 1):
        state = """
        <State> {}
          <Mean> {}
           {}
          <Variance> {}
           {}
        """.format(i + 1, dims, vec(means), dims, vec(variances))
        states.append(state)
        transitions[i,i] = 0.9
        if i + 1 < max_states:
            transitions[i, i + 1] = 0.1
            
    return """
    ~o <VecSize> {} <USER>
    ~h "{}"

    <BeginHMM>
      <NumStates> {}
      {} 
      <TransP> {}
      {}
    <EndHMM>
    """.format(dims, name, max_states, "".join(states), max_states, mat(transitions))


def simple_grammar(label_file):
    df = pd.read_csv(label_file, sep=" ", header=None, names=["start", "stop", "lab"], skiprows=2)
    df = df.dropna()
    labels = list(set(df["lab"]))
    patterns = " | ".join(labels)
    patterns = "$patterns = {};".format(patterns)
    grammars = "( SENT-START (<$patterns>) SENT-END )"
    grammars = "( $patterns )"
    return "\n".join([patterns, grammars])


def wordlist(label_file):
    df = pd.read_csv(label_file, sep=" ", header=None, names=["start", "stop", "lab"], skiprows=2)
    df = df.dropna()
    labels = list(set(df["lab"]))
    labels.append("SENT-START")
    labels.append("SENT-END")
    labels = sorted(labels)
    labels = ["{} {}".format(i,j) for i, j in zip(labels, labels)]
    return "\n".join(labels)


def mmf(label_file, proto_file, dim, hmm_out="hmm0", hmm_list_out="monophones"):
    df = pd.read_csv(label_file, sep=" ", header=None, names=["start", "stop", "lab"], skiprows=2)
    df = df.dropna()
    labels = set(df["lab"])
    print(labels)
    lines = [line for line in open(proto_file)]

    hmm = []
    start = False
    for line in lines:
        if line.startswith("<BEGINHMM>"):
            start = True
        if start:
            hmm.append(line)    
        if line.endswith("<ENDHMM>"):
            break

    monophones = []
    mmf = ["""~o <VECSIZE> {} <USER><DIAGC>""".format(dim)]

    for i in labels:    
        header = "~h \"{}\"\n".format(i)
        monophones.append("{}".format(i))
        mmf.append(header)
        mmf.extend(hmm)
        mmf.append("\n")

    mmf = "".join(mmf)
    monophones = "\n".join(monophones)
    with open(hmm_out, "w") as f:
        f.write(mmf)
    with open(hmm_list_out, "w") as f:
        f.write(monophones)        


def htk_name(num):
    return "".join([str(chr(int(i) + 97)) for i in str(num)])
    

def get_ll(out):
    out = out.split(b"\n")
    for line in out:
        if b"average log prob per frame" in line:
            ll = line.strip().split(b" ")[-1]
            ll = float(ll)
            return ll

        
def take_step(folder, i):
    files = glob.glob("{}/data/train/*.htk".format(folder))
    out = check_output(["rm", "-rf", "{}/hmm{}".format(folder, i)])
    out = check_output(["mkdir", "{}/hmm{}".format(folder, i)])
    out = check_output("HERest -A -T 1 -v {} -I {}/clusters_TRAIN.mlf -M {}/hmm{} -H {}/hmm{}/hmm_mmf {}/list".format(FLOOR, folder, folder, i, folder, i - 1, folder).split(" ") + files)
    return get_ll(out)


def htk_eval(folder, last_hmm):
    files = glob.glob("{}/data/test/*.htk".format(folder))
    out = check_output("HVite -T 1 -n 10 -p 0.0 -s 5.0 -H {}/hmm{}/hmm_mmf -i {}/predictions.mlf -w {}/wdnet {}/dict {}/list".format(
        folder, last_hmm, folder, folder, folder, folder
    ).split(" ") + files)
    out = check_output("HResults -I {}/clusters_TEST.mlf {}/list {}/predictions.mlf".format(folder, folder, folder).split(" "))
    return out.decode("utf-8")
    

def htk_export(folder, out_htk, out_lab, k=5, min_c = 4):
    instances_file   = "{}/instances.pkl".format(folder)
    predictions_file = "{}/predictions.pkl".format(folder)
    clusters_file    = "{}/clusters.pkl".format(folder)
    label_file       = "{}/labels.pkl".format(folder)
    
    instances   = pkl.load(open(instances_file, "rb")) 
    clusters    = pkl.load(open(clusters_file, "rb"))[k, :]
    predictions = pkl.load(open(predictions_file, "rb")) 
    label_dict  = pkl.load(open(label_file, "rb"))
    reverse     = dict([(v,k) for k, v in label_dict.items()])
    
    ids_cluster = {}
    for i, cluster in enumerate(clusters):
        if len(instances[i]) > 0: 
            if cluster not in ids_cluster:
                ids_cluster[cluster] = []
            ids_cluster[cluster].append(i)
        
    cur = 0
    label_dict = {}
    train = "{}_TRAIN.mlf".format(out_lab.replace(".mlf", ""))
    test  = "{}_TEST.mlf".format(out_lab.replace(".mlf", ""))
    os.system("mkdir {}/train".format(out_htk))
    os.system("mkdir {}/test".format(out_htk))
    n_exp = 0
    with open(train, 'w') as fp_train, open(test, 'w') as fp_test:
        fp_train.write("#!MLF!#\n")
        fp_test.write("#!MLF!#\n")
        for c, ids in ids_cluster.items():
            label = label_cluster(predictions, ids, reverse)
            if c not in label_dict:
                label_dict[c] = htk_name(c)
            if label != "ECHO" and len(ids) > min_c:
                n_exp += 1
                random.shuffle(ids)                
                train_ids = ids[0:len(ids)//2]
                test_ids  = ids[len(ids)//2:len(ids)]
                for i in train_ids:
                    n = len(instances[i])
                    write_htk(instances[i], "{}/train/{}_{}.htk".format(out_htk, label_dict[c], i))
                    fp_train.write("\"*/{}_{}.lab\"\n".format(label_dict[c], i))
                    fp_train.write("{} {} {}\n".format(0, n, label_dict[c]))
                    fp_train.write(".\n")
                for i in test_ids:
                    n = len(instances[i])
                    write_htk(instances[i], "{}/test/{}_{}.htk".format(out_htk, label_dict[c], i))
                    fp_test.write("\"*/{}_{}.lab\"\n".format(label_dict[c], i))
                    fp_test.write("{} {} {}\n".format(0, n, label_dict[c]))
                    fp_test.write(".\n")                    
    print("#clusters: {}".format(n_exp))
