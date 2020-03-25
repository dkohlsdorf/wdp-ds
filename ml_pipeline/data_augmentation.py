import numpy as np 
from audio import dataset
from scipy.signal import resample


def subsample(x, by=2):
    '''
    Subsample the whole signal

    :param x:  a time series
    :param by: subsampling factor
    :returns:  subsampled time series
    '''
    return resample(x, int(len(x) / by))


def upsample(x, by=2):
    '''
    Upsample the whole signal

    :param x: a time series
    :param by: upsampling factor
    :returns: subsampled time series
    '''
    return repreat(x, by, axis=0)    


def shrink(x, sec=0.1, sample_rate=44000, by=2):
    '''
    Subsample a random window simulating time warp
    
    :param x: a time series
    :param by: subsampling factor
    :param sec: window size in seconds
    :param sample_rate: signal's sample rate
    :returns: warped time series
    '''
    n   = len(x)
    win = sec * sample_rate
    i = np.random.randint(win, n)
    return np.concatenate([x[0:i-win], subsample(x[i-win:i], by), x[i:n]])


def expand(x, sec=0.1, sample_rate=44000, by=2):
    '''
    Up a random window simulating time warp
    
    :param x: a time series
    :param by: upsampling factor
    :param sec: window size in seconds
    :param sample_rate: signal's sample rate
    :returns: warped time series
    '''
    n   = len(x)
    win = sec * sample_rate
    i = np.random.randint(win, n)
    return np.concatenate([x[0:i-win], upsample(x[i-win:i], by), x[i:n]])


def additive_noise(x, noises):
    '''
    Add some noise

    :param x: a time series
    :param noises: noise snippets
    :returns: time series with noise

    '''
    m  = len(noises)
    n  = len(x)
    i  = np.random.randint(0, m)
    to = min(n, len(noises[i])) 
    return x[:to] + noises[i][:to]


class SpectrogramGenerator:

    def __init__(self, input_folder, params, lable, noises):
        self.input_folder = input_folder
        self.params       = params
        self.lable        = lable
        self.noises       = noises

    def generate(self, sample=5):
        '''
        Generate augmented examples 
        '''
        for (x, y, _, _, _) in dataset(input_folder, params, lable, True):
            for i in range(0, sample):
                noise = np.random.uniform() > 0.5
                _x = None
                if i == 0:
                    _x = x 
                elif i % 2 == 0:
                    by       = random.randint(2, 5)
                    do_upsample = np.random.uniform() > 0.5
                    if do_upsample:
                        _x = upsample(x, by)
                    else:
                        _x = subsample(x, by)
                else:
                    by        = random.randint(2, 3)
                    do_expand = np.random.uniform() > 0.5
                    if do_expand:
                        _x = expand(x, by=by)
                    else:
                        _x = shrink(x, by=by)
                if noise:
                        _x = additive_noise(_x, self.noises)
                yield _x, y