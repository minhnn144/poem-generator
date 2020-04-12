from lib.vnlp import vnlp
from lib.data_utils import data_utils
from tqdm import tqdm
inp_len = 25
out_len = 28
att_len = 5
att_text = "Lãng mạn, Hiện đại, Tiếc nuối"

padd = ("<PAD>", [0]*300)

nlp = vnlp.VNlp()
nlp.from_bin("/model/baomoi.vn.300.model")
# nlp.add_vector(padd[0], padd[1])
# nlp.to_bin("/model/baomoi.vn.300.model")
inp = []
att = []
out = []
with open('./raws/data.txt', 'r') as f:
    data = f.read().split("\n\n\n")
    att_toks = att_text.lower().split(", ")
    att_toks = nlp.fill_sequence(att_toks[:att_len], padd[0], att_len)
    att_vecs = [nlp.to_vector(t) for t in att_toks]
    for i in tqdm(range(len(data))):
        inp_toks = nlp.get_token(nlp.normalization(data[i]), POS=["N"])
        inp_toks = list(set(inp_toks))
        inp_toks = nlp.fill_sequence(inp_toks[:inp_len], padd[0], inp_len)
        inp_vecs = [nlp.to_vector(t) for t in inp_toks]
        inp.append(inp_vecs)

        att.append(att_vecs)

        out_toks = nlp.normalization(data[i]).split()
        if len(out_toks) < out_len:
            print("miss match")
            out_toks = nlp.fill_sequence(out_toks, padd[0], out_len)
        out_vecs = [nlp.to_vector(w) for w in out_toks][:out_len]
        out.append(out_vecs)

data_utils.pickle_data('/data/vectorized/input.pkl', inp)
data_utils.pickle_data('/data/vectorized/att.pkl', att)
data_utils.pickle_data('/data/vectorized/output.pkl', out)
