import unittest
from lib.data_utils import data_utils
import numpy as np

f = ['output.pkl', 'input.pkl', 'att.pkl']
p = "./data/vectorized/"

class verify_data(unittest.TestCase):
    def test_size(self):
        for _ in f:
            data = data_utils.unpickle_file(p + _, full=True)
            l1 = len(data)
            _len = len(data[0])
            for i1 in range(l1):
                l2 = len(data[i1])
                self.assertEqual(l2, _len, "Size miss match! {}: {}".format(l2, _len))
                for i2 in range(l2):
                    l3 = len(data[i2])
                    self.assertEqual(l3, l2, "Size miss match! {}: {}".format(l3, l2))         
    
    def test_data_type(self):
        for _ in f:
            data = data_utils.unpickle_file(p + _, full=True)
            for i1 in range(len(data)):
                for i2 in range(len(data[i1])):
                    for i3 in range(len(data[i1][i2])):
                        self.assertIs(type(data[i1][i2][i3]), np.float32, "Wrong data type!")

    def test_data_length(self):
        len_ = set()
        for _ in f:
            data = data_utils.unpickle_file(p + _, full=True)
            len_.add(len(data))
        self.assertEqual(len(len_), 1, "Data len not equal")