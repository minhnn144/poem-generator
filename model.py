import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, inp_size, hid_size, drop_rate):
        super(Encoder, self).__init__()
        self.Dropout = nn.Dropout(drop_rate)
        self.BiLSTM = nn.LSTM(input_size=inp_size,
                              hidden_size=hid_size, num_layers=3)

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
        _out = self.Linear(out)
        out = _out.reshape(out.size(0), out.size(1), -1)
        return out

class Generator(nn.Module):
    def __init__(self, word_size, embed_size, hid_size, drop_rate=0.2):
        super(Generator, self).__init__()
        # self.Encoder = Encoder(embed_size, hid_size, drop_rate)
        self.Embedding = nn.Embedding(word_size, embed_size)
        self.Decoder = Decoder(embed_size, hid_size, word_size, drop_rate)

    def forward(self, inp):
        out = self.Embedding(inp)
        out = self.Decoder(out)
        return out