# Author: Nathan Trouvain at 10/11/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import argparse

from pmseq.canarygan.inception.train import train

parser = argparse.ArgumentParser(
    description="Distributed canaryGAN inception scorer training loop. "
    "Powered by Ligthning and Pytorch."
)

parser.add_argument(
    "--save_dir", type=str, help="Model directory (checkpoints, logs...)."
)
parser.add_argument("--data_dir", type=str, help="Dataset root directory.")
parser.add_argument(
    "--max_epochs", type=int, default=20, help="Max number of training epochs."
)
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
parser.add_argument(
    "-G", "--devices", type=int, default=1, help="Devices per node used for training."
)
parser.add_argument(
    "-N", "--num_nodes", type=int, default=1, help="Nodes used for training."
)
parser.add_argument(
    "-c", "--num_workers", type=int, default=12, help="Data loader processes."
)
parser.add_argument(
    "--log_every_n_steps",
    type=int,
    default=50,
    help="Training steps between 2 Tensorboard summaries.",
)
parser.add_argument(
    "--save_topk", type=int, default=5, help="Min rank of best saved checkpoints."
)
parser.add_argument(
    "--save_every_n_epochs", type=int, default=1, help="Epochs between 2 checkpoints."
)
parser.add_argument("--seed", type=int, default=None, help="Fix all random generators.")
parser.add_argument(
    "--resume", action="store_true", help="If set, resume training to last checkpoint."
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    help="If set, will save checkpoints and metrics in a 'scratch' directory. "
    "'scratch' will be emptied before start.",
)
parser.add_argument(
    "--early_stopping",
    action="store_true",
    help="If set, will stop training when validation accuracy reaches a plateau or "
    "decreases.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    train(**vars(args))
