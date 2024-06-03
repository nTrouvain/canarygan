# Author: Nathan Trouvain at 8/31/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import shutil

import lightning as pl

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils import data

from .model import CanaryGAN
from ..utils import prepare_checkpoints
from ..dataset import Canary16kDataset


def train(
    save_dir,
    data_dir,
    max_epochs=1000,
    batch_size=64,
    devices=1,
    num_nodes=1,
    num_workers=12,
    log_every_n_steps=100,
    save_every_n_epochs=15,
    save_topk=5,
    resume=True,
    seed=0,
    version="infer",
    dry_run=False,
):
    """
    Train a CanaryGAN instance.

    This function may optimally be used in a distributed setup,
    as described in Lightning documentation. It should however also
    work locally, by setting "num_nodes=1" and "devices=1".

    Parameters
    ----------
    save_dir : str or Path
        Directory where weights checkpoints and training logs will
        be saved.
    data_dir : str or Path
        Path to Canary 16k syllables dataset.
    max_epochs : int, default to 1000
        Maximum number of training epochs.
    batch_size : int, default to 64
        Training batch size, per node. If training in a
        distributed setup, each node will receive a batch of
        size batch_size.
    devices : int, default to 1
        Number of training devices per node. If each node has
        two GPUs, then devices=2.
    num_nodes : int, default to 1
        Number of compute nodes allocated for this run. If num_nodes=2,
        then 2 nodes out of all allocated compute nodes will be used
        for training this instance. See Lightning documentation for
        more information on how to perform distributed training.
    num_workers : int, default to 12,
        Number of processes to attach to each device to load data.
    log_every_n_steps : int, default to 100
        Metrics logging period, in number of training steps (batches).
        Logs are stored as Tensorboard log files and CSV files.
    save_every_n_epochs : int, default to 15
        Checkpoint period, in number of epochs.
    save_topk : int, default to 5
        Number of checkpoints to keep based on best discriminator loss.
        (this does not work very well, as discriminator loss is not a
        valid indicator of GAN generation quality).
    resume : bool, default to True
        Restart training from last available checkpoint.
    seed : int, default to 0
        Random state seed. Note that perfect reproducibility is not
        ensured.
    version : str or int or "infer", default to "infer"
        Instance ID. If "infer", will increment version number automatically
        based on previous runs stored in save_dir.
    dry_run : bool, default to False
        If True, save results in a scratch directory. This directory will be
        overwritten if another dry run is launched.
    """
    pl.seed_everything(seed, workers=True)

    save_dir = pathlib.Path(save_dir)
    data_dir = pathlib.Path(data_dir)

    ckpt_dir, tb_dir, csv_dir, curr_version = prepare_checkpoints(
        save_dir=save_dir, version=version, dry_run=dry_run, resume=resume
    )

    dataset = data.DataLoader(
        Canary16kDataset(data_dir),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    logger = TensorBoardLogger(
        tb_dir,
        name="",
        log_graph=False,
        version=curr_version,
    )
    csv_logger = CSVLogger(csv_dir, name="", version=curr_version)

    cktp_filename = (
        "epoch_{epoch:03d}-step_{step:05d}-disc_loss_epoch_{disc_loss_epoch:.3f}"
    )

    epoch_cktpointer = ModelCheckpoint(
        dirpath=ckpt_dir / "all",
        monitor="epoch",
        mode="max",
        filename="all_" + cktp_filename,
        every_n_epochs=save_every_n_epochs,
        save_top_k=-1,  # keep them all
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,  # avoid '=' characters in name
    )
    last_cktpointer = ModelCheckpoint(
        dirpath=ckpt_dir / "last",
        filename="latest_" + cktp_filename,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    topk_cktpointer = ModelCheckpoint(
        dirpath=ckpt_dir / f"top{save_topk}",
        filename="top_" + cktp_filename,
        monitor="disc_loss_epoch",
        mode="max",  # as we monitor critic loss
        save_top_k=save_topk,
        every_n_epochs=1,
        # we don't need to whole training state, just the best weights
        save_weights_only=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )

    model = CanaryGAN(seed=seed)

    # Note that this part may need to be adapted depending on the distributed
    # infrastucture you are using.
    trainer = pl.Trainer(
        accelerator="auto",
        devices=devices,
        num_nodes=num_nodes,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=max_epochs,
        logger=[logger, csv_logger],
        log_every_n_steps=log_every_n_steps,
        callbacks=[
            epoch_cktpointer,
            topk_cktpointer,
            last_cktpointer,
            RichModelSummary(max_depth=3),
            RichProgressBar(leave=True),
        ],
        # Remove forced determinism, because some operations do not support it (yet).
        # See error bellow:
        # RuntimeError: reflection_pad1d_backward_out_cuda
        # does not have a deterministic implementation,
        # but you set 'torch.use_deterministic_algorithms(True)'.
        # You can turn off determinism just for this operation,
        # or you can use the 'warn_only=True' option,
        # if that's acceptable for your application.
        # deterministic=True,
    )

    last_ckpt = ckpt_dir / "last" / "last.ckpt"

    if resume:
        if last_ckpt.is_file():
            trainer.fit(model, dataset, ckpt_path=str(last_ckpt))
        else:
            raise FileNotFoundError(f"No checkpoint to resume. {last_ckpt} not found.")
    else:
        trainer.fit(model, dataset)
