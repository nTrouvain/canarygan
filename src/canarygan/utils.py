# Author: Nathan Trouvain at 29/08/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import shutil

import torch


def select_device(device="cpu", device_idx=0):
    """Select device based on device type and index.
    By default, select CPU.
    """
    if device == "cpu":
        return device

    if torch.cuda.device_count() > 1:
        device = f"{device}:{device_idx}"
    elif torch.cuda.is_available():
        device = "cuda"

    return device


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
