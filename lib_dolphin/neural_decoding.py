import numpy as np
from scipy.signal import butter, lfilter

MIN_LEN = 44100 // 4
MAX_LEN = 44100 // 2

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def draw_noise(length, noise):
    N = len(noise)
    start  = np.random.randint(length, N - length)
    sample = noise[start: start + length]
    return sample


def draw_signal(df, signals):    
    row = df.sample()    
    start = row['starts'].iloc[0]
    stop  = row['stops'].iloc[0]
    return signals[start:stop]


def combined(length, df, signals, noise):
    noise = np.concatenate([draw_noise(length, noise) for i in range(0, 10)])
    N = len(noise)        
    signal = butter_bandpass_filter(draw_signal(df, signals), 1000, 15000, 44100)
    n = int(min(len(signal), N))

    start_w = np.arange(0, int(n * 0.001)) / int(n * 0.001)
    stop_w  = np.flip(start_w)

    if N-n > n:
        insert_at = np.random.randint(n, N-n)
        signal[:int(n * 0.001)] *= start_w
        signal[n-int(n * 0.001):] *= stop_w
    else:
        insert_at = 0
    p = np.random.uniform()
    noise_p = 1.0 - p
    noise[insert_at:insert_at+n] = noise_p * noise[insert_at:insert_at+n] + p * signal[0:n]
    return noise, insert_at, insert_at+n