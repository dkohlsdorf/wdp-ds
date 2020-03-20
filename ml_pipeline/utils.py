import pickle

def unpickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1') 
