# Author: Nathan Trouvain at 10/17/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import argparse

from pmseq.canarygan.inception.score import inception_score, baseline_inception_score


parser = argparse.ArgumentParser(
    description="Distributed canaryGAN inception scorer training loop. "
                "Powered by Ligthning and Pytorch."
)

parser.add_argument(
    "exp",
    type=str,
    choices=["baseline", "gan"],
    help="Experiment to run: compute baseline IS from dataset "
    "(baseline) or compute IS or GAN generator (gan)."
)

parser.add_argument(
    "--scorer_ckpt", type=str, help="Model directory (checkpoints, logs...)."
)
parser.add_argument(
    "--data_dir", type=str, default=None, help="Dataset root directory."
)
parser.add_argument(
    "--generator_version", type=str, default=None, help="Dataset root directory."
)
parser.add_argument(
    "--vec_dir", type=str, default=None,
)
parser.add_argument(
    "--out_dir", type=str
)
parser.add_argument(
    "--n_samples", type=int, default=50000, help="Max number of training epochs."
)
parser.add_argument(
    "--latent_dim", type=int, default=3,
)
parser.add_argument(
    "--n_split", type=int, default=10
)
parser.add_argument(
    "--batch_size", type=int, default=100, help="Training batch size."
)
parser.add_argument(
    "-G", "--devices", type=int, default=1, help="Devices per node used for training."
)
parser.add_argument(
    "-N", "--num_nodes", type=int, default=1, help="Nodes used for training."
)
parser.add_argument(
    "-c", "--num_workers", type=int, default=12, help="Data loader processes."
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.exp == "gan":
        inception_score(**vars(args))
    else:
        baseline_inception_score(**vars(args))
