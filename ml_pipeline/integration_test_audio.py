import unittest

from audio import * 
from audio_collection import *

class AudioTests:

    FILE = 'ml_pipeline/tests/test.wav'
    OUT  = 'data/test.wav'

    def test_audio_read(self):
        f  = read(AudioTests.FILE)
        collection = AudioSnippetCollection(AudioTests.OUT)
        collection.write(f)
        collection.close()
        f2 = read(AudioTests.OUT)
        diff = np.sum(f2[:len(f)] - f)
        assert diff == 0
        
tests = AudioTests()
tests.test_audio_read()