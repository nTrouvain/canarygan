import click

from ...gan import train

_epilog = """Note : This function may optimally be used in a distributed setup,
as described in Lightning documentation. It should however also
work locally, by setting "--num_nodes=1" and "--devices=1" (default).

Example Usage :

\tLaunch training :

\t\tcanarygan train-gan -s /my/checkpoints -d /my/dataset 

\tResume training from last checkpoint of version 1 :

\t\tcanarygan train-gan -s /my/checkpoints -d /my/dataset \\
\n\t\t--resume \\
\n\t\t--version 1

\tWhile runnning on a cluster with 1 allocated node and 2 GPU :
        
\t\tcanarygan train-gan -s /my/checkpoints -d /my/dataset -N 2 -G 2
"""


@click.command(name="train-gan", help="Train a CanaryGAN instance.", epilog=_epilog)
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(),
    help="Model directory (checkpoints, logs...)",
)
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Dataset root directory.",
)
@click.option(
    "--max-epochs",
    "max_epochs",
    type=int,
    default=1000,
    show_default=True,
    help="Max number of training epochs.",
)
@click.option(
    "--batch_size", type=int, default=64, show_default=True, help="Training batch size."
)
@click.option(
    "-G",
    "--devices",
    type=int,
    default=1,
    show_default=True,
    help="Devices per node used for training.",
)
@click.option(
    "-N",
    "--num_nodes",
    type=int,
    default=1,
    show_default=True,
    help="Nodes used for training.",
)
@click.option(
    "-c",
    "--num_workers",
    type=int,
    default=12,
    show_default=True,
    help="Data loader processes.",
)
@click.option(
    "--log-every-n-steps",
    type=int,
    default=50,
    show_default=True,
    help="Training steps between 2 Tensorboard summaries.",
)
@click.option(
    "--save-every-n-epochs",
    type=int,
    default=15,
    show_default=True,
    help="Epochs between 2 checkpoints.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    show_default=True,
    help="Fix all random generators.",
)
@click.option(
    "--resume",
    is_flag=True,
    show_default=True,
    help="If set, resume training to last checkpoint.",
)
@click.option(
    "--version",
    type=str,
    default="infer",
    show_default=True,
    help="GAN version identifier. Number with auto increment by default.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    show_default=True,
    help="If set, will save checkpoints and metrics in a 'scratch' directory. "
    "'scratch' will be emptied before start.",
)
def train_gan(*args, **kwargs):
    train(*args, **kwargs)
