import torch
from torch import nn
from poem_dataset import PoemDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim.adam import Adam

class Encoder(nn.Module):
    def __init__(self, inp_size, hid_size):
        super(Encoder, self).__init__()
        self.BiLSTM = nn.GRU(inp_size, hid_size, num_layers=3)
        self.Linear = nn.Linear(hid_size, hid_size)

    def forward(self, inp):
        out, hid = self.BiLSTM(inp)
        out = self.Linear(out)
        out = torch.tanh(out)
        return out, hid


class Decoder(nn.Module):
    def __init__(self, inp_size, hid_size, window_size, word_size):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.GRU = nn.GRU(inp_size, hid_size, num_layers=3, batch_first=True)
        self.Linear = nn.Linear(hid_size, 1)
        self.Combiner = nn.Linear(window_size, word_size)

    def forward(self, inp):
        out, hid = self.GRU(inp)
        _out = out.reshape(-1, out.size(2))
        _out = self.Linear(_out)
        _out = _out.reshape(-1, self.window_size)
        _out = self.Combiner(_out)
        out = torch.sigmoid(_out)
        return out

class PoemGeneratorModel(nn.Module):
    def __init__(self, word_size, embed_size, hid_size) -> None:
        super().__init__()
        self.Encoder = Encoder(embed_size, hid_size)
        self.sentiment = nn.Linear(embed_size, hid_size)
        self.scale = nn.Linear(hid_size * 2, hid_size)
        self.Decoder = Decoder(hid_size, hid_size, 10, word_size)
    
    def forward(self, input, sen):
        out, hid = self.Encoder(input)
        out = self.Decoder(out)
        return out

class PoemGeneratorLightning(pl.LightningModule):
    def __init__(self, word_size, embed_size, hid_size, seq_len, batch_size=32, lr=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PoemGeneratorModel(word_size, embed_size, hid_size)
        self.word_size = word_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.hid_size = hid_size
        self.batch_size = batch_size
        self.lr = lr
    
    def forward(self, inp, att):
        return self.model(inp, att)

    def setup(self, stage: str):
        inp_ = "/data/vectorized/inp.pkl"
        ctr_ = "/data/vectorized/sen.pkl"
        out_ = "/data/vectorized/out.pkl"
        dataset = PoemDataset(inp_, ctr_, out_)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_set, self.val_set = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        inp, att, out = batch
        predict = self(inp, att)
        loss = F.cross_entropy(predict, out)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, att, out = batch
        predict = self(inp, att)
        val_loss = F.cross_entropy(predict, out)
        self.log('val_loss', val_loss)