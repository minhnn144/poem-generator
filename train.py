import math
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import os
from tqdm import tqdm
import re
from lib.data_utils import data_utils
from model import Generator
import matplotlib.pyplot as plt


device = "cuda:0" if torch.cuda.is_available() else "cpu"

embed_size = 300
word_size = 2717
hid_size = 200
seq_len = 35
chkp_path = './model/model.chkp'
lr = 0.0001
epochs = 100
batch = 10
def train(epochs, batch, model, criterion, optimizer, inp, out, plot=False):
    model.train()
    if plot:
        loss_ = []
    for e in tqdm(range(epochs)):
        for i in range(len(inp)):
            optimizer.zero_grad()
            predict = model(inp[i])
            predict = predict.reshape(-1, word_size)
            tar = out[i].view(-1)
            loss = criterion(predict, tar)
            if plot and i % 10 == 0:
                loss_.append(loss.item())

            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), chkp_path)
    print("checkpoint saved!")
    if plot:
        plt.plot(loss_)
        plt.show()

model = Generator(word_size, embed_size, hid_size).to(device)
if os.path.isfile(chkp_path):
    model.load_state_dict(torch.load(chkp_path))
    print("checkpoint loaded!")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

inp = data_utils.unpickle_file("/data/vectorized/inp_idx.pkl")
out = data_utils.unpickle_file("/data/vectorized/out_idx.pkl")

data_len = math.floor(len(inp) / batch) * batch 

print("data loaded!")
inp = torch.tensor(inp[:data_len]).view(-1, batch, seq_len).to(device)
out = torch.tensor(out[:data_len]).view(-1, batch, seq_len).to(device)

train(epochs, batch, model, criterion, optimizer, inp, out, plot=True)