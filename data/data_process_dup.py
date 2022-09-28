from lib import data_utils, vnlp
from lib.dictionary import Dictionary
import re
from tqdm import tqdm
import logging
import warnings


word_vec_size = 300
embed_size = 301
hid_size = 200
seq_len = 35

config_path = "/config.ini"

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
logging.info('Setting up dictionary and nlp module')
dictionary = Dictionary(vector_size=embed_size)
nlp = vnlp.VNlp('../model/wiki.vi.bin')
control_toks = ["<SOS>", "<EOS>", "<PAD>", "<BRK>"]
for tok in control_toks:
    dictionary.add_word(tok, nlp.to_vector(tok))

blacklist_token = "[\W_]+"
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
logging.info("Saving configuration")
config = {
    "hid_size": hid_size,
    "emb_size": embed_size,
    "word_size": len(dictionary),
    "seq_len": seq_len
}
data_utils.save_config(config_path, config)