import unittest
import numpy as np

from ml_pipeline.genetic_programming import *

class GeneticProgrammingTest(unittest.TestCase):

    def test_gp(self):
        BRACKET = 0
        SEQ     = 1

        cost = {
            REPEAT     : 2.0,
            AND_SYMBOL : 1.0,
            OR_SYMBOL  : 5.0,
            DONT_CARE  : 10.0 
        }

        examples = [
            [1,0,0,1],
            [1,0,0,0,1],
            [1,0,0,0,0,1],
            [1,0,0,0,0,0,1],
            [0,0,0, 1],
            [0,0,1,1],
            [0,0,1],
            [0,0,0,0,0,0,1,1,1,1,1]
        ]

        labels = [
            BRACKET, BRACKET, BRACKET, BRACKET,
            SEQ, SEQ, SEQ, SEQ
        ]

        e, s, l = evolve(examples, labels, cost)
        x = [(tree, score, label) for tree, score, label in zip(e, s, l)]
        x.sort(key=lambda x: -x[1])
        df = pd.DataFrame(data=[[e, s, l] for e, s, l in x[0:10]], columns=["expression", "score", "label"])
        print(df.head())
        self.assertTrue(str(df['expression'][0]) ==  "(0)+(1)+" or str(df['expression'][0]) ==  "1(0)+1")
        self.assertTrue(str(df['expression'][1]) ==  "(0)+(1)+" or str(df['expression'][1]) ==  "1(0)+1")