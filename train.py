import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from lib.data_utils import data_utils
import os
from tqdm import tqdm
from model import GNet

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
    att_len = 5
    word_size = 300
    learning_rate = 0.001
    chkp_path = "./model/train.chkp"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("{} has been used".format(device))
    Generator = GNet(inp_len, out_len, word_size).to(device)
    if os.path.isfile(chkp_path):
        Generator.load_state_dict(torch.load(chkp_path))
        print("checkpoint loaded")
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
