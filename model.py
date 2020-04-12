import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size, seq_len, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hid_size = hid_size
        self.Dropout = nn.Dropout(dropout)
        self.BiLSTM = nn.LSTM(input_size=in_size, hidden_size=hid_size,
                              num_layers=num_layers, batch_first=True, bidirectional=True)
        self.Linear = nn.Linear(in_features=seq_len, out_features=out_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inp):
        out = self.Dropout(inp)
        out, (hid, cel) = self.BiLSTM(out)
        out = out.reshape(-1, self.hid_size * 2, out.size(1))
        out = self.Linear(out)
        out = self.Sigmoid(out)
        out = out.reshape(-1, out.size(-1), self.hid_size * 2)
        return out, hid, cel


class Attention(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=2):
        super(Attention, self).__init__()
        self.BiLSTM = nn.LSTM(input_size=in_size, hidden_size=hid_size,
                              num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, inp):
        _, (hid, cel) = self.BiLSTM(inp)
        out = torch.cat([hid[0], hid[1]], dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hid_size = hid_size
        self.Dropout = nn.Dropout(dropout)
        self.BiLSTM = nn.LSTM(input_size=in_size, hidden_size=hid_size,
                              num_layers=num_layers, batch_first=True, bidirectional=True)
        self.Linear = nn.Linear(in_features=hid_size *
                                2, out_features=out_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inp, hid, cel):
        out = self.Dropout(inp)
        out, (hid, cel) = self.BiLSTM(out, (hid, cel))
        out = self.Linear(out)
        out = self.Sigmoid(out)
        return out


class GNet(nn.Module):
    def __init__(self, in_len, out_len, word_size):
        super(GNet, self).__init__()
        self.out_len = out_len
        self.Encoder = Encoder(
            word_size, 400, out_size=out_len, seq_len=in_len, num_layers=4, dropout=0.2)
        self.Attention = Attention(word_size, 100)
        self.Decoder = Decoder(1000, 400, word_size, 4, dropout=0.2)

    def forward(self, inp, att):
        en, hid, cel = self.Encoder(inp)
        en = en.reshape(en.size(0) * en.size(1), -1)
        att = self.Attention(att)
        att = att.reshape(att.size(0), -1)
        att = att.repeat(self.out_len, 1)
        en_att = torch.cat([att, en], dim=1)
        en_att = en_att.reshape(-1, self.out_len, en_att.size(-1))
        out = self.Decoder(en_att, hid, cel)
        return out


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

    def forward(self, *input, **kwargs):
        return super().forward(*input, **kwargs)
