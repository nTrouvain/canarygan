# Author: Nathan Trouvain at 10/17/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import argparse

from pmseq.decoder.score import score


parser = argparse.ArgumentParser(
    description="Score generator using RNN decoder."
)

parser.add_argument(
    "--decoder_ckpt", type=str, help="Decoder model checkpoint."
)
parser.add_argument(
    "--vec_file", type=str, default=None, help="Latent vectors on disk (.npy)"
)
parser.add_argument(
    "--generator_version", type=str, default=None, help="Dataset root directory."
)
parser.add_argument(
    "--save_dir", type=str
)
parser.add_argument(
    "--n_is_split", type=int, default=10
)
parser.add_argument(
    "--batch_size", type=int, default=100, help="Training batch size."
)
parser.add_argument(
    "-c", "--num_workers", type=int, default=12, help="Data loader processes."
)
parser.add_argument(
    "--device_idx", type=int, default=0, help="GPU device index."
)

if __name__ == "__main__":
    args = parser.parse_args()
    score(**vars(args))
