import pickle as pkl

def pkl_load(path):
    with open(path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p