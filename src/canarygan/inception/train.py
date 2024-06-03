# Author: Nathan Trouvain at 10/11/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import warnings

import lightning as pl

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from .model import LightningSyllableClassifier
from ..dataset import Canary16kDataModule
from ..utils import prepare_checkpoints


def train(
    save_dir,
    data_dir,
    max_epochs=1000,
    batch_size=64,
    devices=1,
    num_nodes=1,
    num_workers=12,
    log_every_n_steps=100,
    save_every_n_epochs=1,
    save_topk=5,
    seed=4862,
    resume=False,
    dry_run=False,
    early_stopping=False,
):
    """
    Train a syllable classifier for Inception Score computation.

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
    save_every_n_epochs : int, default to 1
        Checkpoint period, in number of epochs.
    save_topk : int, default to 5
        Number of checkpoints to keep based on best validation accuracy.
    resume : bool, default to True
        Restart training from last available checkpoint.
    seed : int, default to 0
        Random state seed. Note that perfect reproducibility is not
        ensured.
    dry_run : bool, default to False
        If True, save results in a scratch directory. This directory will be
        overwritten if another dry run is launched.
    early_stopping : bool, default to False
        If True, enable early stopping based on validation accuracy.
        Patience is set to 100 steps and minimum accuracy delta to 0.001.
    """
    pl.seed_everything(seed, workers=True)

    save_dir = pathlib.Path(save_dir)
    data_dir = pathlib.Path(data_dir)

    ckpt_dir, tb_dir, csv_dir, curr_version = prepare_checkpoints(
        save_dir,
        dry_run=dry_run,
        resume=resume,
    )

    datamodule = Canary16kDataModule(
        data_dir,
        batch_size=batch_size,
        with_classes=True,
        num_workers=num_workers,
        split=True,
    )

    model = LightningSyllableClassifier(seed=seed)

    logger = TensorBoardLogger(
        tb_dir,
        name="",
        log_graph=False,
        version=curr_version,
    )
    csv_logger = CSVLogger(csv_dir, name="", version=curr_version)

    cktp_filename = (
        "epoch_{epoch:03d}-step_{step:05d}-"
        "loss_{loss_epoch:.3f}-val_acc_{val_acc:.2f}"
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
        monitor="val_acc",
        mode="max",  # as we monitor validation accuracy
        save_top_k=save_topk,
        every_n_epochs=1,
        # we don't need to whole training state, just the best weights
        save_weights_only=True,
        auto_insert_metric_name=False,
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

    callbacks = [
        topk_cktpointer,
        last_cktpointer,
        epoch_cktpointer,
        RichModelSummary(max_depth=3),
        RichProgressBar(leave=False),
    ]

    if early_stopping:
        early_stopper = EarlyStopping(
            monitor="val_acc",
            min_delta=0.001,
            patience=100,
            mode="max",
            check_on_train_epoch_end=False,
        )
        callbacks.append(early_stopper)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=devices,
        num_nodes=num_nodes,
        strategy="auto",
        max_epochs=max_epochs,
        logger=[logger, csv_logger],
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
    )

    tester = pl.Trainer(
        accelerator="auto",
        devices=1,  # Lightning warning: test on only 1 device to avoid duplicata
        num_nodes=1,
        strategy="auto",
        logger=[logger, csv_logger],
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
    )

    last_ckpt = ckpt_dir / "last" / "last.ckpt"

    if resume:
        if last_ckpt.is_file():
            trainer.fit(model, datamodule=datamodule, ckpt_path=str(last_ckpt))
            tester.test(model, datamodule=datamodule, ckpt_path="best")
        else:
            warnings.warn(f"No checkpoint to resume. {last_ckpt} not found.")
    else:
        trainer.fit(model, datamodule=datamodule)
        tester.test(model, datamodule=datamodule, ckpt_path="best")
