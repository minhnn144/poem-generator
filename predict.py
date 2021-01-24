import torch
import argparse
from lib import data_utils, vnlp
from lib.dictionary import Dictionary
from pytorch_lightning.core.lightning import LightningModule
from model import PoemGeneratorLightning

parser = argparse.ArgumentParser()

parser.add_argument("--config", default="/config.ini")
parser.add_argument("--chkp", default="lightning_logs/version_100/checkpoints/epoch=999-step=999.ckpt")
parser.add_argument("--random", action="store_true")

args = parser.parse_args()

config = data_utils.load_config(args.config)

embed_size = config.getint('DEFAULT', 'emb_size')
word_size = config.getint('DEFAULT', 'word_size')
hid_size = config.getint('DEFAULT', 'hid_size')
seq_len = config.getint('DEFAULT', 'seq_len')

chkp_path = args.chkp

nlp = vnlp.VNlp('model/wiki.vi.bin')
corpus = data_utils.unpickle_file("/data/vectorized/word_list.pkl")
inp_ = torch.tensor(corpus.vectors[corpus.word2idx["<SOS>"]])

if args.random:
    inp_ += 0

sentiments = ["Mưa", "gió", "lạnh"]
sent = nlp.combined_vector(sentiments)

model = PoemGeneratorLightning.load_from_checkpoint(chkp_path, strict=False)
model.freeze()


outputs = []
for i in range(seq_len):
    out = model(inp_, sent)
    out = out.squeeze().exp()
    out = torch.multinomial(out, 1)[0]
    tok = corpus.idx2word[int(out)]
    outputs.append(tok)
    inp_ = out.view(1, -1)
print(len(outputs))
print(" ".join(outputs))