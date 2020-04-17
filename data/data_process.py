from lib.data_utils import data_utils
import re

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)

def main():
    dictionary = Dictionary()

    control_toks = ["<SOS>", "<EOS>", "<PAD>", "<BRK>"]
    for i in control_toks:
        dictionary.add_word(i)

    blacklist_token = "[\W_]+"
    seq_len = 35
    inp_list = []
    out_list = []
    with open("./raws/data.txt", 'r') as f:
        raw = f.read()
        poems = raw.split("\n\n\n")
        for p in poems:
            lines = p.split("\n")
            p_toks = []
            for l in lines:
                l_norm = re.sub(blacklist_token, " ", l.lower())
                l_norm = re.sub("\s+", " ", l_norm)
                l_toks = l_norm.split(" ") + ["<BRK>"]
                for t in l_toks:
                    idx = dictionary.add_word(t)
                    p_toks.append(idx)
            del p_toks[-1]
            inp_toks = [dictionary.word2idx["<SOS>"]] + p_toks
            inp_toks = inp_toks[:seq_len] + [dictionary.word2idx["<PAD>"]] * (seq_len - len(inp_toks))
            out_toks = p_toks + [dictionary.word2idx["<EOS>"]]
            out_toks = out_toks[:seq_len] + [dictionary.word2idx["<PAD>"]] * (seq_len - len(out_toks))
            inp_list.append(inp_toks)
            out_list.append(out_toks)

    data_utils.pickle_data('/data/vectorized/inp_idx.pkl', inp_list)
    data_utils.pickle_data('/data/vectorized/out_idx.pkl', out_list)
    data_utils.pickle_data('/data/vectorized/word_list.pkl', dictionary)
    print(len(inp_list[0]))
    print(len(dictionary))


if __name__ == "__main__":
    main()
