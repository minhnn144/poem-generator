import pytorch_lightning as pl
from lib import data_utils
from model import PoemGeneratorLightning
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import seed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument("--dev", action="store_true")
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--fixed", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--config", type=str, default="/config.ini")
parser.add_argument("--resume", type=str, default="")

args = parser.parse_args()

config = data_utils.load_config(args.config)

embed_size = config.getint('DEFAULT', 'emb_size')
word_size = config.getint('DEFAULT', 'word_size')
hid_size = config.getint('DEFAULT', 'hid_size')
seq_len = config.getint('DEFAULT', 'seq_len')

checkpoint_callback = ModelCheckpoint(filename='{epoch}-{step}.ckpt', save_last=True)

if args.fixed:
    seed.seed_everything(1)

if args.overfit:
    if args.resume != "":
        logging.info('Loading state')
        model = PoemGeneratorLightning.load_from_checkpoint(args.resume, strict=True, word_size=word_size, embed_size=embed_size, hid_size=hid_size, seq_len=seq_len, batch_size=1, lr=args.lr)
    else:
        model = PoemGeneratorLightning(word_size, embed_size, hid_size, seq_len, lr=args.lr, batch_size=1)
    trainer = pl.Trainer(gpus=args.gpu, overfit_batches=1, max_epochs=args.epoch, checkpoint_callback=checkpoint_callback)
    trainer.fit(model)
else:
    if args.resume != "":
        model = PoemGeneratorLightning.load_from_checkpoint(args.resume, word_size=word_size, embed_size=embed_size, hid_size=hid_size, seq_len=seq_len, lr=args.lr, batch_size=args.batch)
    else:
        model = PoemGeneratorLightning(word_size, embed_size, hid_size, seq_len, lr=args.lr, batch_size=args.batch)
    trainer = pl.Trainer(fast_dev_run=args.dev, max_epochs=args.epoch, gpus=args.gpu, checkpoint_callback=checkpoint_callback)
    trainer.fit(model)