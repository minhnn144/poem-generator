import unittest
from lib.vnlp import vnlp
import numpy as np

class TestModel(unittest.TestCase):
    def test_find_vec(self):
        nlp = vnlp.VNlp()
        nlp.from_bin('/model/baomoi.vn.300.model')
        t = "chào"
        v = nlp.to_vector(t)
        print(v)
        key = nlp.nlp.vocab.vectors.most_similar(np.asarray([v]), n=4)
        key = key[0]
        print(key.shape)
        print(nlp.nlp.vocab.strings[5957762358184700636])
        self.assertEqual(t, key, "Shit")

a = TestModel()

a.test_find_vec()