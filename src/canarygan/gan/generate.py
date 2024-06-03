import torch
import numpy as np
import reservoirpy as rpy

from ..decoder.decode import decode 


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
