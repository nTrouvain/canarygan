import re

from pathlib import Path

import numpy as np
import reservoirpy as rpy

from torch.utils import data

from canarygan.dataset import LatentSpaceSamples

from .utils import select_device
from .decoder.decode import maximum_a_posteriori, reduce_time
from .decoder.model import load_decoder
from .gan.model import load_generator
from .gan.generate import generate_and_decode


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

    dataset = LatentSpaceSamples(vec_file, n_samples)

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
