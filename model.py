import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, inp_size, hid_size, drop_rate):
        super(Encoder, self).__init__()
        self.Dropout = nn.Dropout(drop_rate)
        self.BiLSTM = nn.LSTM(inp_size, hid_size, num_layers=3)

    def forward(self, inp):
        out = self.Dropout(inp)
        out, (hid, cel) = self.BiLSTM(out)
        return out, hid, cel


class Decoder(nn.Module):
    def __init__(self, inp_size, hid_size, out_size, drop_rate):
        super(Decoder, self).__init__()
        self.Dropout1 = nn.Dropout(drop_rate)
        self.GRU = nn.GRU(inp_size, hid_size, num_layers=2, batch_first=True)
        self.Dropout2 = nn.Dropout(drop_rate)
        self.Linear = nn.Linear(hid_size, out_size)

    def forward(self, inp):
        out = self.Dropout1(inp)
        out,(hid, cel) = self.GRU(inp)
        out = self.Dropout2(out)
        _out = out.reshape(out.size(0) * out.size(1), out.size(2))
        _out = self.Linear(_out)
        out = _out.reshape(out.size(0), out.size(1), -1)
        out = torch.sigmoid(out)
        return out

class Attention(nn.Module):
    def __init__(self, hid_size):
        super().__init__()
        self.hid_size = hid_size
        self.linear = nn.Linear(self.hid_size * 2, self.hid_size)

    def forward(self, inp, att):
        pass

    def score(self, hidden, encode_outputs):
        inp = torch.cat([hidden, encode_outputs], 2)
        energy = self.linear(inp)
        energy = energy.transpose(1, 2)

class PoemGeneratorModel(nn.Module):
    def __init__(self, word_size, embed_size, hid_size, drop_rate=0.2) -> None:
        super().__init__()
        self.Encoder = Encoder(embed_size, hid_size, drop_rate)
        self.sentiment = nn.Linear(embed_size, hid_size)
        self.Decoder = Decoder(hid_size, hid_size, word_size, drop_rate)
    
    def forward(self, input, sen):
        out, hid, cel = self.Encoder(input)
        sent = self.sentiment(sen)
        sent = torch.sigmoid(sent)
        sent = sent.reshape(out.size(0), -1, sent.size(-1))
        sent = sent.repeat(1, out.size(1), 1)
        out = out + sent
        out = self.Decoder(out)
        return out