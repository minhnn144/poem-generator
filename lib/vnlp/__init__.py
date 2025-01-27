import numpy as np
import fasttext

class VNlp:
    def __init__(self, model):
        self.ft = fasttext.load_model(model)
        self.word_dimension = self.ft.get_dimension()

    def to_vector(self, word: str):
        control_toks = {
            "<SOS>": np.asfarray([0]*300 + [1]),
            "<EOS>": np.asfarray([0]*300 + [2]),
            "<PAD>": np.asfarray([0]*300 + [3]),
            "<BRK>": np.asfarray([0]*300 + [4]),
        }
        word = self.normalize(word)
        if word in control_toks.keys():
            return control_toks[word]
        else:
            return np.append(self.ft.get_word_vector(word), 0.)

    @staticmethod
    def normalize(word: str):
        w = " ".join(word.split())
        return w.lower()

    def combined_vector(self, words: list):
        vecs = []
        for w in words:
            vecs.append(self.to_vector(w))
        vecs = np.asfarray(vecs)
        return np.mean(vecs, axis=0)

    @staticmethod
    def generate_bow_vector(word_size, index):
        res = [0] * word_size
        res[index] = 1
        return np.array(res, dtype=np.float64)