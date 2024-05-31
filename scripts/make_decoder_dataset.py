import argparse

from pathlib import Path
from pmseq.decoder.dataset import DecoderDataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--features", type=str)
parser.add_argument("--n_mfcc", type=int, default=13)

args = parser.parse_args()

data_dir = Path(args.data_dir)
ckpt_dir = Path(args.ckpt_dir)
dataset = (
    DecoderDataset(data_dir=data_dir, features=args.features, n_mfcc=args.n_mfcc)
    .get()
    .checkpoint(ckpt_dir)
)

print(dataset.train_data[0].shape, dataset.train_data[1].shape)
