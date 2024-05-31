# Author: Nathan Trouvain at 9/15/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import torch

from .train import CanaryGAN


def generate(from_ckpt, output_dir, n_samples):
    ckpt_path = pathlib.Path(from_ckpt)

    # ckpt = torch.load(ckpt_path)
    gan = CanaryGAN.load_from_checkpoint(ckpt_path)
    print()


if __name__ == "__main__":
    generate("./checkpoints/all/last.ckpt", None, None)
