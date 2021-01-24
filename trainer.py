import pytorch_lightning as pl
from lib import data_utils
from model import PoemGeneratorLightning
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dev", action="store_true")
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--config", type=str, default="/config.ini")

args = parser.parse_args()

config = data_utils.load_config(args.config)

embed_size = config.getint('DEFAULT', 'emb_size')
word_size = config.getint('DEFAULT', 'word_size')
hid_size = config.getint('DEFAULT', 'hid_size')
seq_len = config.getint('DEFAULT', 'seq_len')


if args.overfit:
    model = PoemGeneratorLightning(word_size, embed_size, hid_size, seq_len, lr=args.lr, batch_size=1)
    trainer = pl.Trainer(gpus=args.gpu, overfit_batches=1)
    trainer.fit(model)
else:
    model = PoemGeneratorLightning(word_size, embed_size, hid_size, seq_len, lr=args.lr, batch_size=args.batch)
    trainer = pl.Trainer(fast_dev_run=args.dev, max_epochs=args.epoch, gpus=args.gpu)
    trainer.fit(model)