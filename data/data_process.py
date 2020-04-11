from lib.vnlp import vnlp
from lib.data_utils import data_utils
from tqdm import tqdm
inp_len = 20
out_len = 28
att_text = "Tình yêu"

padd = ("<PAD>", [0]*300)

nlp = vnlp.VNlp()
nlp.from_bin("/model/wiki.vi.model")
nlp.add_vector(padd[0], padd[1])

inp = []
att = []
out = []
with open('./raws/data.txt', 'r') as f:
    data = f.read().split("\n\n\n")
    for i in tqdm(range(len(data))):
        inp_toks = nlp.get_token(nlp.normalization(data[i]), POS=["N"])
        inp_toks = list(set(inp_toks))
        inp_toks = nlp.fill_sequence(inp_toks[:inp_len], padd[0], inp_len)
        inp_vecs = [nlp.to_vector(t) for t in inp_toks]
        inp.append(inp_vecs)

        att.append([nlp.to_vector(att_text)])

        out_vecs = [nlp.to_vector(w) for w in data[i].split()][:out_len]
        out.append(out_vecs)

data_utils.pickle_data('/data/vectorized/input.pkl', inp)
data_utils.pickle_data('/data/vectorized/att.pkl', att)
data_utils.pickle_data('/data/vectorized/output.pkl', out)
