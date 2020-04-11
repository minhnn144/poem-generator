import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from lib.data_utils import data_utils
import os
from tqdm import tqdm


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
    def __init__(self, in_size, hid_size, num_layers=1):
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
            word_size, 200, out_size=out_len, seq_len=in_len, num_layers=2, dropout=0.2)
        self.Attention = Attention(word_size, 100)
        self.Decoder = Decoder(600, 200, word_size, 2, dropout=0.2)

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


def train(epochs, batch, model, criterion, optimizer, inp, att, out, chkp_path, plot=False):
    if plot:
        loss_ = []
    for e in tqdm(range(epochs)):
        for i in range(len(inp)):
            optimizer.zero_grad()
            predict = model(inp[i], att[i])
            loss = criterion(predict, out[i])
            if plot and i % 10 == 0:
                loss_.append(loss.item())
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), chkp_path)
    print("checkpoint saved")
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(loss_)
        plt.show()


def main():
    batch = 5
    epochs = 1
    inp_len = 25
    out_len = 28
    att_len = 1
    word_size = 300
    learning_rate = 0.001
    chkp_path = "./model/train.chkp"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("{} has been used".format(device))
    Generator = GNet(inp_len, out_len, word_size).to(device)
    if os.path.isfile(chkp_path):
        Generator.load_state_dict(torch.load(chkp_path))
    inp = data_utils.unpickle_file('/data/vectorized/input.pkl')[:230]
    out = data_utils.unpickle_file('/data/vectorized/output.pkl')[:230]
    att = data_utils.unpickle_file('/data/vectorized/att.pkl')[:230]

    inp = torch.FloatTensor(inp).view(-1, batch, inp_len, word_size).to(device)
    out = torch.FloatTensor(out).view(-1, batch, out_len, word_size).to(device)
    att = torch.FloatTensor(att).view(-1, batch, att_len, word_size).to(device)
    print("data loaded")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Generator.parameters(), lr=learning_rate)
    train(epochs=epochs, batch=batch, model=Generator, criterion=criterion,
          optimizer=optimizer, inp=inp, att=att, out=out, chkp_path=chkp_path, plot=True)
    pass


if __name__ == "__main__":
    main()
