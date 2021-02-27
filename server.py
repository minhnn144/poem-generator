from flask.templating import render_template
import torch
import argparse
from lib import data_utils, vnlp
from lib.dictionary import Dictionary
from model import PoemGeneratorLightning
from flask import Flask, make_response, request, jsonify

parser = argparse.ArgumentParser()

parser.add_argument("--config", default="/config.ini")
parser.add_argument("--chkp", default="lightning_logs/version_0/checkpoints/last.ckpt")

args = parser.parse_args()

config = data_utils.load_config(args.config)

embed_size = config.getint('DEFAULT', 'emb_size')
word_size = config.getint('DEFAULT', 'word_size')
hid_size = config.getint('DEFAULT', 'hid_size')
seq_len = config.getint('DEFAULT', 'seq_len')

chkp_path = args.chkp

nlp = vnlp.VNlp('model/wiki.vi.bin')
corpus = data_utils.unpickle_file("/data/vectorized/word_list.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    inp_ = torch.FloatTensor(corpus.vectors[corpus.word2idx["<SOS>"]])
    inp_ = inp_.reshape(1, 1, -1)

    sentiments = data.get("keywords")
    sent = torch.FloatTensor(nlp.combined_vector(sentiments))
    sent = sent.reshape(1, -1)

    model = PoemGeneratorLightning.load_from_checkpoint(chkp_path, strict=True, word_size=word_size, embed_size=embed_size, hid_size=hid_size, seq_len=seq_len)
    model.eval()
    model.freeze()

    outputs = []
    for i in range(seq_len):
        out = model(inp_, sent)
        out_ = out.squeeze().exp()
        out_ = torch.multinomial(out_, 1)[0]
        tok = corpus.idx2word[int(out_)]
        outputs.append(tok)
        inp_ = torch.FloatTensor(corpus.vectors[int(out_)])
        inp_ = inp_.reshape(1, 1, -1)
    print(len(outputs))
    print(" ".join(outputs))
    return make_response(jsonify({'result': ' '.join(outputs)}), 200)

app.run("0.0.0.0", 1998)