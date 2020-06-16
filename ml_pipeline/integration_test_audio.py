import unittest

from audio import * 
from audio_collection import *

class AudioTests:

    FILE = [
        'ml_pipeline/tests/test.wav',
        'ml_pipeline/tests/test2.ogg'
    ]
    OUT  = [
        'data/test.wav',
        'data/test2.wav'
    ]

    def test_audio_read(self):
        for fpath, opath in zip(AudioTests.FILE, AudioTests.OUT):            
            f  = read(fpath, True)
            collection = AudioSnippetCollection(opath)
            collection.write(f)
            collection.close()
            f2 = read(opath)
            diff = np.sum(f2[:len(f)] - f)
            assert diff == 0
            AudiofileParams.reset()

tests = AudioTests()
tests.test_audio_read()