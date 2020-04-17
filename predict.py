from lib.data_utils import data_utils
from model import Generator
import torch
from data.data_process import Dictionary

device = "cuda:0" if torch.cuda.is_available() else "cpu"

embed_size = 300
word_size = 2048
hid_size = 200
seq_len = 35
chkp_path = "./model/model.chkp"

corpus = data_utils.unpickle_file("/data/vectorized/word_list.pkl")
inp_ = torch.tensor(corpus.word2idx["mưa"])
inp_ = inp_.reshape(1, -1)

model = Generator(word_size, embed_size, hid_size).to(device)

model.load_state_dict(torch.load(chkp_path))

sent = ["mưa"]
for i in range(40):
    out = model(inp_)
    out = out.squeeze().exp()
    out = torch.multinomial(out, 1)[0]
    tok = corpus.idx2word[int(out)]
    if tok == "<EOS>":
        break
    sent.append(tok)
    inp_ = out.view(1, -1)
print(len(sent))
print(" ".join(sent))