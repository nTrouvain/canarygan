"""Command line interface for canarygan."""

import click

from .scripts import sample_z, build_decoder_dataset, train_decoders, train_gan, baseline, gan, train_inception, train_gan, generate_main, umap   


@click.group(name="canarygan", context_settings={"show_default": True})
def cli():
    pass


@cli.group(name="inception", help="Compute inception score.")
def inception_cli():
    pass

inception_cli.add_command(baseline)
inception_cli.add_command(gan)

cli.add_command(train_gan)
cli.add_command(sample_z)
cli.add_command(build_decoder_dataset)
cli.add_command(generate_main)
cli.add_command(train_decoders)
cli.add_command(train_inception)
cli.add_command(umap)
