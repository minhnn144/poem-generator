from model import GNet
import torch
from lib.vnlp import vnlp

inp_sent = "Dưới nắng trưa, dòng sông lấp lánh ánh vàng. Muôn ngàn tia nắng nhảy nhót đùa nghịch trên mặt sông như trẻ nhỏ. Dăm chiếc thuyền đánh cá, tiếng gõ mái chèo đuổi cá dồn dập vang lên giữa khung cảnh tĩnh mịch của trưa hè."
att_sent = "Lãng mạn, Hiện đại, Tiếc nuối"

nlp = vnlp.VNlp()
nlp.from_bin('/model/wiki.vi.model')
pad = ("<PAD>", [0]*300)


def inp_vectorize(sent, max_len):
    toks = nlp.get_token(nlp.normalization(sent), POS=["N"])
    toks = list(set(toks))
    toks = nlp.fill_sequence(toks[:max_len], pad[0], max_len)
    vecs = [nlp.to_vector(t) for t in toks]
    return vecs


def att_vectorize(sent, max_len):
    toks = att_sent.split(", ")
    toks = nlp.fill_sequence(toks[:max_len], pad[0], max_len)
    vecs = [nlp.to_vector(t) for t in toks]
    return vecs


def get_closest(vecs):
    t = [nlp.get_closest(v.data.numpy())[0] for v in vecs]
    return " ".join(t)


def predict(model, inp, att):
    vecs = model(inp, att)
    return vecs[0]


def main():
    inp_len = 25
    att_len = 5
    out_len = 28
    word_size = 300
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    Generator = GNet(inp_len, out_len, word_size).to(device)
    Generator.load_state_dict(torch.load('./model/poem.chkp', map_location=torch.device(device)))
    inp = inp_vectorize(inp_sent, inp_len)
    att = att_vectorize(att_sent, att_len)
    inp = torch.FloatTensor(inp).view(1, inp_len, word_size).to(device)
    att = torch.FloatTensor(att).view(1, att_len, word_size).to(device)
    out = predict(Generator, inp, att)
    print(out.size())
    print(get_closest(out))
    pass


if __name__ == "__main__":
    main()
