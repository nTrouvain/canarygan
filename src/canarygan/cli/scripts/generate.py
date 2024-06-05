import re

from pathlib import Path

import click
import numpy as np
import reservoirpy as rpy
import torch

from torch.utils import data

from canarygan.dataset import LatentSpaceSamples

from ...utils import select_device
from ...decoder import load_decoder
from ...gan import load_generator, generate_and_decode, generate


@click.command(
    "generate",
    help="Generate canary syllables using a trained GAN generator instance. May also decode the generated sounds.",
)
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(),
    required=True,
    help="Directory where results will be saved.",
)
@click.option(
    "-x",
    "--generator-ckpt",
    "generator_instance",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Generator checkpoint file.",
)
@click.option(
    "-z",
    "--vec-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Numpy archive storing latent vectors.",
)
@click.option(
    "-y",
    "--decoder-ckpt",
    "decoder_instances",
    default=[],
    multiple=True,
    help="Decoder checkpoint file. May be used several times to decode with different decoders.",
)
@click.option(
    "-n",
    "--num_samples",
    type=int,
    default=10000,
    help="If no vec-file is provided, will randomly sample n latent vectors.",
)
@click.option(
    "-p",
    "--transform-params",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Preprocessing parameters for decoder. Must be a YAML file storing DecoderDataset parameters.",
)
@click.option("--batch_size", default=100, type=int)
@click.option("-c", "--num_workers", default=8, type=int)
@click.option(
    "--device_idx", default=0, type=int, help="Only useful in multi-GPU context."
)
@click.option("--device", type=str, default="cpu")
def generate_main(
    vec_file,
    save_dir,
    generator_instance,
    decoder_instances,
    n_samples=10000,
    batch_size=100,
    num_workers=8,
    device_idx=0,
    device="cpu",
    transform_params=None,
):
    rpy.verbosity(0)
    np.random.seed(0)

    dataset = LatentSpaceSamples(vec_file, n_samples)

    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = select_device(device, device_idx)
    generator = load_generator(generator_instance, device)

    version = epoch = step = "None"
    if (
        match := re.search(
            r"version_(\d+).*epoch_(\d+).*step_(\d+)", str(generator_instance)
        )
    ) is not None:
        version, epoch, step = match.groups("None")

    predictions = {}
    x_gens = []
    zs = []
    if len(decoder_instances) != 0:
        for instance in decoder_instances:
            decoder = load_decoder(instance)
            name = decoder.__class__.__name__.lower()
            policy = decoder.reduce_time_policy

            click.echo(f"Decoding GAN generation using: {decoder.name}")

            x_gens, y_gens, zs = generate_and_decode(
                decoder,
                generator,
                dataloader,
                device=device,
                save_gen=True,
                transform_params=transform_params,
            )
            predictions[f"y_{name}_{policy}"] = y_gens
    else:
        for z in dataloader:

            x_gen = generate(generator, z, device)
            zs.append(torch.as_tensor(z))
            x_gens.append(x_gen)

        x_gens = np.concatenate(x_gens, axis=0)
        zs = torch.concatenate(zs, dim=0).squeeze().cpu().numpy()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"generation-version_{version}-epoch_{epoch}-step_{step}"

    np.savez_compressed(
        save_dir / f"{filename}.npz",
        x=x_gens,
        z=zs,
        version=version,
        epoch=epoch,
        step=step,
        **predictions,
    )

    return
