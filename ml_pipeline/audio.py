import numpy as np
import random
import os
import tensorflow as tf

from health_checks import *
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
        #AudiofileParams.__instance.check_params(rate, dtype, sample_width)
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


def dataset(folder, params, label_func, shuffle):
    """
    Build an iterator over labeled spectrograms from a folder

    :param folder: the folder we search
    :param params: window parameters
    :param label_func: how to label the instances
    :param shuffle: if we shuffle the windows per file

    :returns: iterator (spectrogram, label, filename, start, stop)
    """
    for filename in tf.io.gfile.listdir(folder):
        if filename.endswith('.ogg') or filename.endswith('.wav') or filename.endswith('.aiff') or filename.startswith('cluster') or filename.startswith('noise'):
            path = "{}/{}".format(folder, filename)
            spec_iter = labeled_spectrogram_windows(path, params, label_func, shuffle=shuffle)
            for x in spec_iter:
                yield x


def labeled_spectrogram_windows(filename, params, label_func, shuffle=False):
    """
    Generate spectrogram windows from file as well as labels
    generated from the filename or spectrogram.

    For example:
      a binary classifier: f(name, x) => 1 if name == 'noise' else 0
      an auto encoder:     f(name, x) => x
    :param filename: the filename
    :param params: parameters
    :param shuffle: shuffle the dataset
    :param label_func: f(filename, spectrogram) => target

    :returns: iterator (spectrogram, label, filename, start, stop)
    """
    for (spectrogram, _, start, stop) in spectrogram_windows(filename, params, shuffle):
        label = label_func(filename, spectrogram)
        yield (spectrogram, label, filename, start, stop)


def spectrogram_windows(filename, params, shuffle=False, pcen=True):
    """
    Extract all spectrogram windows from an audio file.
    Also z-normalizes the spectrograms

    :param params: Windowing parameters
    :param highpass: Frequency below which we cut the spectrogram

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
            if pcen:
                mfcc = librosa.feature.mfcc(audio.astype(np.float), n_mfcc=64)
                e = librosa.pcen(spec, gain=0.2, bias=5)
                e = (e - np.mean(e)) / (np.std(e) + 1)
                yield (e, filename, start, stop) 
            else:
                spec  = fwd_spectrogram(audio, params.fft_win_filtered, params.fft_step)
                dft_start = params.fft_win - params.n_fft_bins
                dft_stop  = params.fft_win
                spec  = spec[:, dft_start:dft_stop]
                mu      = np.mean(spec)
                sigma   = np.std(spec) + 1.0
                yield ((spec - mu) / sigma, filename, start, stop)
    

def spectrogram_regions(filename, params, regions, pcen=True):
    """
    Spectrogram Region Extraction

    :param filename: the filename
    :param params: Windowing parameters
    :param regions: sequence of start, stop tuples
    :param first_channel: pick the first channel for multi channel data or take the average
    :returns: spectrogram of the normalized region
    """
    data = read(filename)

    for (start, stop) in regions:
        audio = data[start:stop]
        spec  = fwd_spectrogram(audio, params.fft_win_filtered, params.fft_step)
        dft_start = params.fft_win - params.n_fft_bins
        dft_stop  = params.fft_win
        spec  = spec[:, dft_start:dft_stop]
        if pcen:
            e = librosa.pcen(spec, gain=0.2, bias=5)
            e = (e - np.mean(e)) / (np.std(e) + 1)
            yield e
        else:
            mu      = np.mean(spec)
            sigma   = np.std(spec) + 1.0
            yield (spec - mu) / sigma
        

def audio_snippets(snippets):
    """
    Extract snippets from audio

    snippets: (start, stop, filename)
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


def fwd_spectrogram(audio, win=512, step=64, print_stats=False):
    """
    Compute the spectrogram of audio data

    :param audio: one channel audio
    :param win: window size for dft sliding window
    :param step: step size for dft sliding windo

    :returns: power spectrum
    """
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = np.abs(fft(audio[i - win: i] * hanning))
        spectrogram.append(dft)
    spectrogram = np.array(spectrogram)
    if print_stats:
        logaudio.info(statistics(spectrogram))
    return spectrogram
