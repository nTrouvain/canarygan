from functools import reduce
from pathlib import Path
import re
from matplotlib.pyplot import axis

import torch
import numpy as np
import reservoirpy as rpy

from torch.utils import data
from rich.progress import track

from .utils import load_decoder, load_generator, load_latent_vects, select_device
from .decoder.decoding import decode, maximum_a_posteriori, reduce_time


def generate(generator, z, device="cpu"):
    if isinstance(z, np.ndarray):
        z = torch.as_tensor(z)

    z = z.to(device)
    z = torch.atleast_2d(z)

    generator.eval()

    with torch.no_grad():
        x_gen = generator(z)

    return x_gen.squeeze().cpu().numpy()


def generate_and_decode(
    decoder,
    generator,
    latent_vects,
    device="cpu",
    save_gen=False,
):
    rpy.verbosity(0)

    y_gens = []
    x_gens = []
    zs = []
    for z in latent_vects:

        x_gen = generate(generator, z, device)

        if save_gen:
            x_gens.append(x_gen)
            zs.append(torch.as_tensor(z))

        y_gen = decode(decoder, x_gen)

        y_gens.append(y_gen)

    y_gens = np.concatenate(y_gens, axis=0)

    if save_gen:
        x_gens = np.concatenate(x_gens, axis=0)
        zs = torch.concatenate(zs, dim=0).squeeze().cpu().numpy()
        return x_gens, y_gens, zs

    return y_gens


def main(
    vec_file,
    save_dir,
    generator_instance,
    decoder_instance,
    n_samples=10000,
    batch_size=100,
    num_workers=8,
    device_idx=0,
    device="cpu",
):
    rpy.verbosity(0)
    np.random.seed(0)

    dataset = load_latent_vects(vec_file, n_samples)

    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = select_device(device, device_idx)

    decoder = load_decoder(decoder_instance)
    idx_to_class = {v: k for k, v in decoder.class_to_idx.items()}

    generator = load_generator(generator_instance, device)

    version = epoch = step = "None"
    if (
        match := re.search(
            r"version_(\d+).*epoch_(\d+).*step_(\d+)", str(generator_instance)
        )
    ) is not None:
        version, epoch, step = match.groups("None")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Unormalized linear predictions (no softmax), all timesteps
    x_gen, y_seq, zs = generate_and_decode(
        decoder, generator, dataloader, device=device, save_gen=True
    )

    predictions = {}
    for policy in ["mean", "max", "argmax", "sum"]:
        y_reduced = reduce_time(y_seq, axis=1, policy=policy)
        y_pred = maximum_a_posteriori(y_reduced, axis=1)
        y_pred = np.array([idx_to_class[y] for y in y_pred])

        predictions[f"y_{policy}"] = y_pred

    filename = f"generation-version_{version}-epoch_{epoch}-step_{step}"

    np.savez_compressed(
        save_dir / f"{filename}.npz",
        x=x_gen,
        y=predictions["y_sum"],
        z=zs,
        version=version,
        epoch=epoch,
        step=step,
        **predictions,
    )

    return
