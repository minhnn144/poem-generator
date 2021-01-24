from lib.data_utils import data_utils
from lib.vnlp import vnlp
import re
import numpy as np
from tqdm import tqdm
import logging
import warnings

class Dictionary:
    def __init__(self, vector_size=300, default_vector=[0]*300) -> None:
        self.vectors = []
        self.word2idx = {}
        self.idx2word = []
        self.vector_size = vector_size
        self.default_vector = default_vector

    def add_word(self, word: str, vector=None):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            if vector is None:
                self.vectors.append(self.default_vector)
            elif vector.size != self.vector_size:
                print("Vector size not match")
                return None
            else:
                self.vectors.append(vector)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def main():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info('Setting up dictionary and nlp module')
    dictionary = Dictionary(vector_size=301)
    nlp = vnlp.VNlp('../model/wiki.vi.bin')
    control_toks = ["<SOS>", "<EOS>", "<PAD>", "<BRK>"]
    for tok in control_toks:
        dictionary.add_word(tok, nlp.to_vector(tok))

    blacklist_token = "[\W_]+"
    seq_len = 35
    inp_list = []
    ctr_list = []
    out_list = []
    data_path = "./raws/input.txt"
    logging.info('Loading raw data at {}'.format(data_path))
    with open(data_path, 'r') as f:
        raw = f.read()
        poems = raw.split("\n\n\n")
        for block in tqdm(poems):
            part = block.split("\n\n")

            ctrs = part[1].split(',')
            ctr_list.append(nlp.combined_vector(ctrs))

            p = part[0]
            lines = p.split("\n")
            p_toks = []
            p_vecs = []
            for l in lines:
                l_norm = re.sub(blacklist_token, " ", l.lower())
                l_norm = re.sub("\s+", " ", l_norm)
                l_toks = l_norm.split(" ") + ["<BRK>"]
                for t in l_toks:
                    idx = dictionary.add_word(t, nlp.to_vector(t))
                    p_toks.append(idx)
                    p_vecs.append(dictionary.vectors[idx])
            del p_toks[-1]
            del p_vecs[-1]

            inp_toks = [dictionary.vectors[dictionary.word2idx["<SOS>"]]] + p_vecs
            inp_toks = inp_toks[:seq_len] + \
                [dictionary.vectors[dictionary.word2idx["<PAD>"]]] * (seq_len - len(inp_toks))
            out_toks = p_toks + [dictionary.word2idx["<EOS>"]]
            out_toks = out_toks[:seq_len] + \
                [dictionary.word2idx["<PAD>"]] * (seq_len - len(out_toks))
            inp_list.append(inp_toks)
            out_list.append(out_toks)
    logging.info("Saving pre-processed data to file")
    data_utils.pickle_data('/data/vectorized/inp_idx.pkl', inp_list)
    data_utils.pickle_data('/data/vectorized/ctr_idx.pkl', ctr_list)
    data_utils.pickle_data('/data/vectorized/out_idx.pkl', out_list)
    data_utils.pickle_data('/data/vectorized/word_list.pkl', dictionary)
    
    logging.info("Vocabulary size: {}".format(len(dictionary)))


if __name__ == "__main__":
    main()
