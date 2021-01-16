import numpy as np
import random
import os
import tensorflow as tf

from dtw import * 

import librosa
import logging

logging.basicConfig()
logaudio = logging.getLogger('audio')
logaudio.setLevel(logging.INFO)

from pydub import AudioSegment

from numpy.fft import fft
from collections import namedtuple


class AudiofileParams(object):
    """
    Singleton saving audio file parameters globally
    """

    __instance = None

    def __new__(cls, rate, dtype, sample_width):
        if AudiofileParams.__instance is None:
            AudiofileParams.__instance = object.__new__(cls)
            AudiofileParams.__instance.rate = rate
            AudiofileParams.__instance.dtype = dtype
            AudiofileParams.__instance.sample_width = sample_width
        return AudiofileParams.__instance

    @classmethod
    def reset(cls):
        AudiofileParams.__instance  = None

    @classmethod
    def get(cls):
        return AudiofileParams.__instance

    @classmethod
    def set_rate(cls, rate):
        AudiofileParams.__instance.rate = rate

    @classmethod
    def set_dtype(cls, dtype):
        AudiofileParams.__instance.dtype = dtype

    @classmethod
    def set_sample_width(cls, sample_width):
        AudiofileParams.__instance.sample_width = sample_width

    def check_params(self, rate, dtype, sample_width):
        if self.rate != rate:
            logaudio.info('Sample rate does not match: {} != {}'.format(self.rate, rate))
        if self.dtype != dtype:
            logaudio.info('Sample type does not match: {} != {}'.format(self.dtype, dtype))
        if self.sample_width != sample_width:
            logaudio.info('Sample width does not match: {} != {}'.format(self.sample_width, sample_width))


class WindowParams(namedtuple('WindowParams', 'spec_win spec_step fft_win fft_step highpass')):

    @property
    def n_fft_bins(self):
        return self.fft_win // 2

    @property
    def fft_win_filtered(self):
        """
        Pad the FFT window by the highpass so the target dimension stays the same
        """
        return self.fft_win + 2 * self.highpass

    @property
    def win_len(self):
        """
        Length of the spectrogram window returned in audio samples
        """
        return self.spec_win * self.fft_step + self.fft_win_filtered

    @property
    def step(self):
        """
        Step size of spectrogram window in audio samples
        """
        return self.spec_step * self.fft_step

    def range(self, sample):
        """
        The range of a sample in a spectrogam window in the audio file

        :param sample: a sample in the spectrogram window

        :returns: start sample in raw audio, stop sample in raw audio
        """
        start_audio = sample * self.step
        stop_audio  = start_audio + self.win_len
        return start_audio, stop_audio

    def len(self, audio_samples):
        """
        Compute the number of spectrogram window in an audio file

        :param audio_samples: number of audio samples in audio_file

        :returns: length of the window
        """
        return (audio_samples - self.win_len) // self.step + 1


def read(path, first_channel=True):
    """
    Read an audio file from local file system or cloud storage
    all all formats that ffmpeg supports are supported

    :param path: pathlike object
    :param first_channel: if true take the first channel for multi channel audio, if false average all channels 
    :returns: audio file with shape (time, )
    """
    with tf.io.gfile.GFile(path, "rb") as f:
        try:
            x = AudioSegment.from_file(f)
            rate = x.frame_rate
            sample_width = x.sample_width
            raw = x.get_array_of_samples()
            x   = np.array(raw).reshape(int(x.frame_count()), x.channels)
            dtype = x.dtype
            params = AudiofileParams(rate, dtype, sample_width)
            if len(x.shape) > 1 and not first_channel:
                x = np.mean(x, axis=1)
            elif len(x.shape) > 1 and first_channel:
                x = x[:, 0]
            else:
                x = x.reshape((len(x)))

        except Exception as e:
            logaudio.info("Skip file {} = {}".format(path, e))
            x = None
    return x


def dataset(folder, params, shuffle):
    """
    Build an iterator over labeled spectrograms from a folder

    :param folder: the folder we search
    :param params: window parameters
    :param shuffle: if we shuffle the windows per file

    :returns: iterator (spectrogram, label, filename, start, stop)
    """
    f = [filename for filename in tf.io.gfile.listdir(folder) if filename.endswith('.ogg') or filename.endswith('.wav') or filename.endswith('.aiff') or filename.startswith('cluster') or filename.startswith('noise')]
    ordered = [i for i in range(0, len(f))]
    if shuffle:
        random.shuffle(ordered)
    for i in ordered:
        filename  = f[i]
        path = "{}/{}".format(folder, filename)
        spec_iter = spectrogram_windows(path, params, shuffle=shuffle)
        for x in spec_iter:
            yield x


def encode(data, enc):
    '''
    Encode the data

    :param data: a sequence [(vec, filename, start, stop) ... ]
    :param enc: encoder neural network
    :returns: matrix with encoded sequence
    '''
    x = np.stack([x.reshape(x.shape[0], x.shape[1], 1) for (x,_,_,_) in data])
    h = enc.predict(x)
    mu_h  = np.mean(h, axis=1)
    std_h = np.std(h, axis=1) 
    h     = ((h.T - mu_h) / std_h).T
    return h


def alignments(folder, params, encoder):
    '''
    Compute all matches between all sequences with distance

    :param folder: folder with audio files
    :param params: windowing parameters
    :param encoder:encoder neural network
    :returns: [(vector, filename, start, stop) ... ], [(spec_i, spec_j, ti, tj, distance) ... ]
    '''

    f = [filename for filename in tf.io.gfile.listdir(folder) if filename.endswith('.ogg') or filename.endswith('.wav') or filename.endswith('.aiff') or filename.startswith('cluster') or filename.startswith('noise')]
    n = len(f)

    spectrograms = []
    for i in range(0, n):
        data   = [x for x in spectrogram_windows(f[i], params)]   
        stds   = [np.std(x) for (x,_,_,_) in data]
        std_th = np.percentile(stds, 75)
        data   = [d for d, std in zip(data, stds) if std > std_th]
        spectrograms.append(data)

    indices = []
    for spectrogram in spectrograms:
        index = [(f, start, stop) for (_, f, start, stop) in spectrogram]
        indices.append(index)

    matches = []
    vectors = {}
    for i in range(0, n):
        x     = encode(spectrograms[i], encoder)
        t,f,_ = x.shape 
        x = x.reshape((t, f))
        for j in range(i + 1, n):
            y = encode(spectrograms[j], encoder)
            t,f,_ = y.shape 
            y = y.reshape((t, f))
            w = int(max(len(x), len(y)) / 10)
            for ti, tj, d in dtw(x, y, w):
                matches.append(i, j, ti, tj, d)
                if (i, ti) not in matches:
                    f, start, stop = indices[i][ti]
                    vectors[(i, ti)] = (x[ti], f, start, stop)
                if (j, tj) not in matches:
                    f, start, stop = indices[j][tj]
                    vectors[(j, tj)] = (y[tj], f, start, stop)
    return vectors, matches


def spectrogram_windows(filename, params, shuffle=False, pcen=False):
    """
    Extract all spectrogram windows from an audio file.
    Also z-normalizes the spectrograms

    :param params: Windowing parameters
    :param shuffle: shuffle the dataset
    :param pcen: use per channel energy normalization?
    :returns: iterator (spectrogram, filename, start, stop)
    """
    assert isinstance(params, WindowParams)
    data = read(filename)
    if data is not None:
        n = len(data)
        n_windows = params.len(n)
        ordered   = [i for i in range(0, n_windows)]
        if shuffle:
            random.shuffle(ordered)
        for i in ordered:
            start, stop = params.range(i)
            audio = data[start:stop]
            spec  = fwd_spectrogram(audio, params.fft_win_filtered, params.fft_step)
            dft_start = params.fft_win - params.n_fft_bins
            dft_stop  = params.fft_win
            spec  = spec[:, dft_start:dft_stop]
            if pcen:
                e = librosa.pcen(spec.T, gain=0.2, bias=5).T
                e = (e - np.mean(e)) / (np.std(e) + 1)
                yield (e, filename, start, stop) 
            else:
                mu      = np.mean(spec)
                sigma   = np.std(spec) + 1.0
                yield ((spec - mu) / sigma, filename, start, stop)
            

def audio_snippets(snippets):
    """
    Extract snippets from audio

    :param snippets: (start, stop, filename)
    """
    files = {}
    for snippet in snippets:
        start = snippet[0]
        stop  = snippet[1]
        filename = snippet[2]
        if filename not in files:
            files[filename] = []
        files[filename].append((start, stop))
    for f, regions in files.items():
        for region in audio_regions(f, regions):
            yield region


def audio_regions(filename, regions):
    """
    Audio Region Extraction

    :param filename: the filename
    :param regions: sequence of start, stop tuples
    :returns: audio snippets
    """
    data = read(filename)
    if data is not None:
        for (start, stop) in regions:
            yield data[start:stop]


def fwd_spectrogram(audio, win=512, step=64):
    """
    Compute the spectrogram of audio data

    :param audio: one channel audio
    :param win: window size for dft sliding window
    :param step: step size for dft sliding windo
    :param print_stats: print debug statistics?
    :returns: power spectrum
    """
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = np.abs(fft(audio[i - win: i] * hanning))
        spectrogram.append(dft)
    spectrogram = np.array(spectrogram)
    return spectrogram
