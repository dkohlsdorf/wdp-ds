import numpy as np

def statistics(spectrogram):
    '''
    Statistics in each dft band of a spectrogram
    
    :param spectrogram: the spectrogram
    :returns: string with mean, standard deviation, 5th percentile and 95th percentile
    '''
    mu  = np.mean(spectrogram, axis=0)
    std = np.std(spectrogram, axis=0)
    p95 = np.percentile(spectrogram, 95, axis=0)
    p05 = np.percentile(spectrogram,  5, axis=0)
    return "Statistics: mu = {}, std = {}, p05 = {}, p95 = {}".format(mu, std, p05, p95)

