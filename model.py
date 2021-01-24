import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, inp_size, hid_size):
        super(Encoder, self).__init__()
        self.BiLSTM = nn.LSTM(inp_size, hid_size, num_layers=1, bidirectional=True)
        self.Linear = nn.Linear(hid_size * 2, hid_size)

    def forward(self, inp):
        out, (hid, cel) = self.BiLSTM(inp)
        out = self.Linear(out)
        out = torch.tanh(out)
        return out, hid, cel


class Decoder(nn.Module):
    def __init__(self, inp_size, hid_size, out_size):
        super(Decoder, self).__init__()
        self.GRU = nn.GRU(inp_size, hid_size, num_layers=1, batch_first=True)
        self.Linear = nn.Linear(hid_size, out_size)

    def forward(self, inp):
        out, hid = self.GRU(inp)
        _out = out.reshape(out.size(0) * out.size(1), out.size(2))
        _out = self.Linear(_out)
        out = _out.reshape(out.size(0), out.size(1), -1)
        out = torch.relu(out)
        return out

class PoemGeneratorModel(nn.Module):
    def __init__(self, word_size, embed_size, hid_size) -> None:
        super().__init__()
        self.Encoder = Encoder(embed_size, hid_size)
        self.sentiment = nn.Linear(embed_size, hid_size)
        self.scale = nn.Linear(hid_size*2, hid_size)
        self.Decoder = Decoder(hid_size, hid_size, word_size)
    
    def forward(self, input, sen):
        out, hid, cel = self.Encoder(input)
        sent = self.sentiment(sen)
        sent = torch.sigmoid(sent)
        sent = sent.reshape(out.size(0), -1, sent.size(-1))
        sent = sent.repeat(1, out.size(1), 1)
        out = torch.cat([out, sent], dim=2)
        out = self.scale(out)
        out = self.Decoder(out)
        return out