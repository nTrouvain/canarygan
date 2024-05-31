# Author: Nathan Trouvain at 8/31/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import argparse

from ..canarygan.train_lightning import train

parser = argparse.ArgumentParser()

parser.add_argument("-N", "--num_nodes", type=int, default=1)
parser.add_argument("-c", "--num_workers", type=int, default=1)
parser.add_argument("-G", "--gpu_devices", type=int, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--log_freq", type=int, default=1)


if __name__ == "__main__":
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        devices=args.gpu_devices,
        num_nodes=args.num_nodes,
        num_workers=args.num_workers,
        batch_size=args.batchsize,
        log_every_n_steps=args.log_freq,
    )
