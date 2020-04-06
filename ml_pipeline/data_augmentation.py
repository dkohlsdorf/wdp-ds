import numpy as np 
from audio import dataset, WindowParams
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
    return np.repeat(x, by, axis=0)    


def shrink(x, samples = 20, by=2):
    '''
    Subsample a random window simulating time warp
    
    :param x: a time series
    :param by: subsampling factor
    :param samples: number of samples
    :returns: warped time series
    '''
    n   = len(x)
    i = np.random.randint(samples, n)
    return np.concatenate([x[0:i-samples], subsample(x[i-samples:i], by), x[i:n]])


def expand(x, samples = 20, by=2):
    '''
    Up a random window simulating time warp
    
    :param x: a time series
    :param by: upsampling factor
    :param samples: number of samples
    :returns: warped time series
    '''
    n   = len(x)
    i = np.random.randint(samples, n)
    return np.concatenate([x[0:i-samples], upsample(x[i-samples:i], by), x[i:n]])


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

    def __init__(self, input_folder, params, noises):
        self.input_folder = input_folder
        self.params       = params
        self.noises       = noises

    def generate(self, sample=5):
        '''
        Generate augmented examples 
        '''
        window = self.params.spec_win * 2 + 1
        params  =  WindowParams(window, self.params.spec_step, self.params.fft_win, self.params.fft_step, self.params.highpass)

        for (x, _, _, _, _) in dataset(self.input_folder, params, lambda f,x: None, True):
            for i in range(0, sample):
                _x = None
                meta = 'original'
                if i == 0:
                    _x = x 
                elif i % 2 == 0:
                    do_upsample = np.random.uniform() > 0.5
                    if do_upsample:
                        by = np.random.randint(2, 5)
                        meta = 'upsample'
                        _x = upsample(x, by)
                    else:
                        meta = 'downsample'
                        _x = subsample(x)
                else:
                    by        = np.random.randint(2, 3)
                    do_expand = np.random.uniform() > 0.5
                    if do_expand:
                        meta = 'expand_region'
                        _x = expand(x, by=by)
                    else:
                        meta = 'shrink_region'
                        _x = shrink(x, by=by)
                for i in range(self.params.spec_win, len(_x), self.params.spec_step):
                    noise  = np.random.uniform() > 0.25
                    noise &= len(self.noises) > 0
                    if noise:
                        yield additive_noise(_x[i-self.params.spec_win:i], self.noises), additive_noise(_x[i-self.params.spec_win:i], self.noises), meta + '_add_noise'
                    else:
                        yield _x[i-self.params.spec_win:i], additive_noise(_x[i-self.params.spec_win:i], self.noises), meta
