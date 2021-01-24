import torch
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.optim.adam import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from model import PoemGeneratorModel
from poem_dataset import PoemDataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dev", action="store_true")
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.0001)

args = parser.parse_args()

class PoemGeneratorLightning(pl.LightningModule):
    def __init__(self, word_size, embed_size, hid_size, batch_size=32, lr=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PoemGeneratorModel(word_size, embed_size, hid_size)
        self.batch_size = batch_size
        self.lr = lr
    
    def forward(self, inp, att):
        return self.model(inp, att)

    def setup(self, stage: str):
        inp_ = "/data/vectorized/inp_idx.pkl"
        ctr_ = "/data/vectorized/ctr_idx.pkl"
        out_ = "/data/vectorized/out_idx.pkl"
        dataset = PoemDataset(inp_, ctr_, out_)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_set, self.val_set = random_split(dataset, [train_size, test_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        inp, att, out = batch
        predict = self(inp, att)
        predict = predict.reshape(-1, word_size, seq_len)
        loss = F.cross_entropy(predict, out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


embed_size = 301
word_size = 1524
hid_size = 200
seq_len = 35


if args.overfit:
    model = PoemGeneratorLightning(word_size, embed_size, hid_size, lr=args.lr, batch_size=1)
    trainer = pl.Trainer(gpus=args.gpu, overfit_batches=1)
    trainer.fit(model)
else:
    model = PoemGeneratorLightning(word_size, embed_size, hid_size, lr=args.lr, batch_size=args.batch)
    trainer = pl.Trainer(fast_dev_run=args.dev, max_epochs=args.epoch, gpus=args.gpu)
    trainer.fit(model)