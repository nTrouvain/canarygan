# Author: Nathan Trouvain <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""Functions for easy syllable generation and decoding."""
import torch
import numpy as np
import reservoirpy as rpy

from ..decoder.decode import decode


def generate(generator, latent_vects, device="cpu"):
    """Generate some syllables.

    Parameters
    ----------
    generator : canarygan.gan.CanaryGANGenerator
        A syllable generator instance.
    latent_vects : np.ndarray or torch.Tensor
        A batch of latent vectors to feed the generator, of shape
        (samples, latent dim).
    device : str, default to "cpu"
        Device on which to perform computations.

    Returns
    -------
    np.ndarray
        Array of shape (samples, audio length). Generated audio.
    """
    if isinstance(latent_vects, np.ndarray):
        latent_vects = torch.as_tensor(latent_vects).float()

    z = latent_vects.to(device)
    z = torch.atleast_2d(z)

    generator.to(device)
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
    transform_params=None,
):
    """Generate and label canary syllables.

    Parameters
    ----------
    decoder : canarygan.decoder.Decoder
        A syllable decoder instance.
    generator : canarygan.gan.CanaryGANGenerator
        A syllable generator instance.
    latent_vects : np.ndarray or torch.Tensor
        A batch or several batches of latent vectors to feed the generator, of shape
        (samples, latent dim) or (samples, batch size, latent dim).
    device : str, default to "cpu"
        Device on which to perform computations.
    save_gen : bool, default to False
        If True, function will return generated audio and corresponding latent vectors
        alongside audio labels. Otherwise, will only return decoded labels.
    transform_params : dict or str, optional
        Preprocessing parameters. If str, will consider it as a path
        and load the file as a canarygan.decoder.dataset.Dataset parameter
        checkpoint file in YAML format. If None, will use
        canarygan.decoder.dataset.Dataset default values.

    Returns
    -------
    If save_gen is True:
    np.ndarray, np.ndarray, np.ndarray
        Audios, decoded labels and latent vectors arrays.
    Else:
    np.ndarray
        Decoded labels.
    """
    rpy.verbosity(0)

    y_gens = []
    x_gens = []
    zs = []

    if isinstance(latent_vects, (np.ndarray, torch.Tensor)):
        # Add batch dimension
        if latent_vects.ndim < 3:
            latent_vects = latent_vects.reshape(1, -1, latent_vects.shape[-1])

    for z in latent_vects:

        x_gen = generate(generator, z, device)

        if save_gen:
            x_gens.append(np.atleast_2d(x_gen))
            zs.append(torch.as_tensor(z))

        y_gen = decode(decoder, np.atleast_2d(x_gen), transform_params)

        y_gens.append(y_gen)

    y_gens = np.concatenate(np.atleast_2d(y_gens), axis=0)

    if save_gen:
        x_gens = np.concatenate(x_gens, axis=0)
        zs = torch.concatenate(zs, dim=0).squeeze().cpu().numpy()
        return x_gens, y_gens, zs

    return y_gens
