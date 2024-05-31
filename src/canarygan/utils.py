# Author: Nathan Trouvain at 29/08/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import shutil

import numpy as np
import torch


def same_padding(input_len, stride, kernel_len):
    """
    "Same" padding mode for Pytorch.

    Adapted from https://www.tensorflow.org/api_docs/python/tf/nn
    """
    if input_len % stride == 0:
        return int(np.ceil(max(kernel_len - stride, 0) / 2))
    else:
        return int(np.ceil(max(kernel_len - (input_len % stride), 0) / 2))


def freeze(module):
    """
    Freeze all module parameters, preventing further training.
    """
    for p in module.parameters():
        p.requires_grad = False
    return module


def unfreeze(module):
    """
    Allow gradient collection for all module parameters.
    """
    for p in module.parameters():
        p.requires_grad = True
    return module


def interpolate(real_x, fake_x):
    """
    Random interpolation of real and generated data,
    as required by the training policy defined in 
    Donahue et al. (2018).
    """
    alpha = torch.rand_like(real_x)
    diff = fake_x - real_x
    interp_x = real_x + (alpha * diff)
    interp_x.requires_grad = True  # Important! used for gradient penalty
    return interp_x


def prepare_checkpoints(save_dir, version="infer", dry_run=False, resume=False):
    """
    Create subdirectories in save_dir to save checkpoints and logs.

    Parameters
    ----------
    save_dir : str or Path
        Checkpoint and log root directory.
    version : str or int or "infer", default to "infer"
        Instance ID. If "infer", will increment version number automatically
        based on previous runs stored in save_dir.
    dry_run : bool, default to False
        If True, save results in a scratch directory. This directory will be 
        overwritten if another dry run is launched.
    resume : bool, default to True
        Use existing checkpoint and log directory to resume training of
        last version.
    """
    if dry_run:
        ckpt_dir = save_dir / "checkpoints" / "scratch"
        tb_dir = save_dir / "logs" / "scratch" / "tensorboard"
        csv_dir = save_dir / "logs" / "scratch" / "csv"
        for d in [ckpt_dir, tb_dir, csv_dir]:
            if d.exists():
                try:
                    shutil.rmtree(d)
                finally:
                    pass
        curr_version = "scratch"
    elif version == "infer":
        ckpt_dir = save_dir / "checkpoints"
        versions = [
            int(n.stem.split("_")[-1])
            for n in sorted(ckpt_dir.iterdir())
            if "version" in str(n)
        ]

        if len(versions) == 0:
            curr_version = "version_0"
        else:
            if resume:
                curr_version = f"version_{max(versions)}"
            else:
                curr_version = f"version_{max(versions) + 1}"

        ckpt_dir = ckpt_dir / curr_version
        tb_dir = save_dir / "logs" / "tensorboard"
        csv_dir = save_dir / "logs" / "csv"
    else:
        if str(version).isdigit():
            curr_version = f"version_{version}"
        else:
            curr_version = version

        ckpt_dir = save_dir / "checkpoints" / curr_version
        tb_dir = save_dir / "logs" / "tensorboard"
        csv_dir = save_dir / "logs" / "csv"

    return ckpt_dir, tb_dir, csv_dir, curr_version
