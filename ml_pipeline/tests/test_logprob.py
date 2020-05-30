import unittest

from math import log
from ml_pipeline.logprob import LogProb, ZERO


class LogProbTest(unittest.TestCase):

    def test_gt(self):
        lx = LogProb.from_float(1.0)
        ly = LogProb.from_float(0.0)
        self.assertTrue(lx > ly)

    def test_from_float(self):
        x  = 0.5
        lx = LogProb.from_float(x)
        ly = LogProb.from_float(0.0)
        self.assertAlmostEqual(lx.prob, log(x), delta=1e-8)
        self.assertTrue(ly.is_zero)
    
    def test_mul(self):
        x = LogProb.from_float(0.5)
        y = LogProb.from_float(0.0)
        self.assertAlmostEqual((x * x).prob, log(0.25), delta=1e-8)
        self.assertAlmostEqual((x * y).prob, ZERO,      delta=1e-8)
        self.assertAlmostEqual((y * x).prob, ZERO,      delta=1e-8)

    def test_add(self):
        x = LogProb.from_float(0.5)
        y = LogProb.from_float(0.0)
        self.assertAlmostEqual((x + x).prob, log(1.0), delta=1e-8)
        self.assertAlmostEqual((x + y).prob, log(0.5), delta=1e-8)
        self.assertAlmostEqual((y + x).prob, log(0.5), delta=1e-8)

    def test_div(self):
        x = LogProb.from_float(1.0)
        y = LogProb.from_float(4.0)
        z = LogProb.from_float(0.0)
        self.assertAlmostEqual((y / y).prob, log(1.0),  delta=1e-8)
        self.assertAlmostEqual((x / y).prob, log(0.25), delta=1e-8)
        self.assertAlmostEqual((y / x).prob, log(4.0),  delta=1e-8)
        self.assertAlmostEqual((z / x).prob, ZERO,      delta=1e-8)


if __name__ == '__main__':
    unittest.main()
