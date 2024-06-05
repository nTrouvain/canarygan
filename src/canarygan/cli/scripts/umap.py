import click

from ...viz import plot_generation_umap


_help = """Make many plots displaying UMAP projections of real and generated syllables.
Will also save UMAP projections and cluster labels in generated syllables metadata.
"""


@click.command("umap", help=_help)
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Dataset directory.",
)
@click.option(
    "-g",
    "--gen-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Generated data directory.",
)
@click.option(
    "--reducer-ckpt",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a trained and saved UMAP object.",
)
@click.option("--epoch", type=int, required=True, help="GAN instance to consider.")
@click.option("--version", type=int, required=True, help="Training epoch to consider.")
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(),
    required=True,
    help="Figures output directory.",
)
def umap(*args, **kwargs):
    plot_generation_umap(*args, **kwargs)
    return
