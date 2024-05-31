# Author: Nathan Trouvain at 10/11/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
from typing import Any
import warnings

import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from .model import SyllableClassifier
from ..dataset import Canary16kDataModule
from ..utils import prepare_checkpoints


class LightningSyllableClassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes=16,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=128,
        f_min=40,
        f_max=7800,
        n_mels=128,
        pad_mode="constant",
        center=True,
        lr=1e-4,
        seed=4862,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.classifier = SyllableClassifier(
            n_classes=n_classes,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            pad_mode=pad_mode,
            center=center,
        )

        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )

        self.train_f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=n_classes,
        )
        self.val_f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=n_classes,
        )

        self.test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.test_f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=n_classes,
        )

    def forward(self, x):
        y_pred = self.classifier(x)
        return torch.argmax(F.softmax(y_pred, dim=1), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier(x)
        loss = F.cross_entropy(logits, y)
        y_pred = torch.argmax(logits, dim=1)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.train_accuracy(y_pred, y)
        self.train_f1_score(y_pred, y)

        # No need to sync_dist, torchmetrics handles the sync for us
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train_f1", self.train_f1_score, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        logits = self.classifier(x_val)
        loss = F.cross_entropy(logits, y_val)
        y_pred = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.val_accuracy(y_pred, y_val)
        self.val_f1_score(y_pred, y_val)

        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_f1", self.val_f1_score, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x_test, y_test = batch
        logits = self.classifier(x_test)
        loss = F.cross_entropy(logits, y_test)
        y_pred = torch.argmax(logits, dim=1)

        self.log("test_loss", loss, sync_dist=True)

        self.test_accuracy(y_pred, y_test)
        self.test_f1_score(y_pred, y_test)

        self.log("test_acc", self.test_accuracy)
        self.log("test_f1", self.test_f1_score)

    def predict_step(self, batch, batch_idx) -> Any:
        return self.classifier(batch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.SGD(self.classifier.parameters(), lr=lr)
        return opt

 
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
