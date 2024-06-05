import click

from ...inception import baseline_inception_score, inception_score, train


@click.command("baseline", help="Compute baseline Inception Score from dataset.")
@click.option(
    "-m",
    "--model-ckpt",
    "scorer_ckpt",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Inception model checkpoint file.",
)
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Dataset root directory.",
)
@click.option(
    "-s",
    "--save_dir",
    "out_dir",
    type=click.Path(),
    required=True,
    help="Results directory.",
)
@click.option(
    "--n-split",
    type=int,
    default=10,
    help="Number of data splits to infer score variance.",
)
@click.option("--batch-size", type=int, default=100)
@click.option(
    "-G", "--devices", type=int, default=1, help="Devices per node used for training."
)
@click.option("-N", "--num-nodes", type=int, default=1, help="Nodes used for training.")
@click.option(
    "-c", "--num-workers", type=int, default=12, help="Data loader processes."
)
@click.option("--seed", type=int, default=0, help="Random state seed.")
def baseline(*args, **kwargs):
    baseline_inception_score(*args, **kwargs)
    return


@click.command("gan", help="Compute GAN Inception Score from generated sounds.")
@click.option(
    "-m",
    "--model-ckpt",
    "scorer_ckpt",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Inception model checkpoint file.",
)
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Dataset root directory.",
)
@click.option(
    "--generator-ckpt",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Generator checkpoint file.",
)
@click.option(
    "-s",
    "--save_dir",
    "out_dir",
    type=click.Path(),
    required=True,
    help="Results directory.",
)
@click.option(
    "--vec-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Numpy archive storing latent vectors.",
)
@click.option(
    "--n_samples",
    type=int,
    default=50000,
    help="If no --vec-file is provided, number of random latent vector sample.",
)
@click.option("--latent_dim", type=int, default=3, help="GAN latent dimension.")
@click.option(
    "--n-split",
    type=int,
    default=10,
    help="Number of data splits to infer score variance.",
)
@click.option("--batch-size", type=int, default=100)
@click.option(
    "-G", "--devices", type=int, default=1, help="Devices per node used for training."
)
@click.option("-N", "--num-nodes", type=int, default=1, help="Nodes used for training.")
@click.option(
    "-c", "--num-workers", type=int, default=12, help="Data loader processes."
)
@click.option("--seed", type=int, default=0, help="Random state seed.")
def gan(*args, **kwargs):
    inception_score(*args, **kwargs)
    return


@click.command(
    "train-inception", help="Distributed canaryGAN inception scorer training loop."
)
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(),
    required=True,
    help="Model directory (checkpoints, logs...).",
)
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Dataset root directory.",
)
@click.option(
    "--max-epochs", type=int, default=20, help="Max number of training epochs."
)
@click.option("--batch-size", type=int, default=64, help="Training batch size.")
@click.option(
    "-G", "--devices", type=int, default=1, help="Devices per node used for training."
)
@click.option("-N", "--num-nodes", type=int, default=1, help="Nodes used for training.")
@click.option(
    "-c", "--num-workers", type=int, default=12, help="Data loader processes."
)
@click.option(
    "--log-every-n-steps",
    type=int,
    default=50,
    help="Training steps between 2 Tensorboard summaries.",
)
@click.option(
    "--save-topk", type=int, default=5, help="Min rank of best saved checkpoints."
)
@click.option(
    "--save-every-n-epochs", type=int, default=1, help="Epochs between 2 checkpoints."
)
@click.option("--seed", type=int, default=None, help="Fix all random generators.")
@click.option(
    "--resume", is_flag=True, help="If set, resume training to last checkpoint."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="If set, will save checkpoints and metrics in a 'scratch' directory. "
    "'scratch' will be emptied before start.",
)
@click.option(
    "--early-stopping",
    is_flag=True,
    help="If set, will stop training when validation accuracy reaches a plateau or "
    "decreases.",
)
def train_inception(*args, **kwargs):
    train(*args, **kwargs)
