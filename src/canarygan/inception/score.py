# Author: Nathan Trouvain at 10/16/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import yaml
from pathlib import Path

import lightning as pl
import torch
import numpy as np

from torch.utils import data
from torchmetrics.image.inception import InceptionScore

from ..gan.model import CanaryGAN
from ..dataset import Canary16kDataset, LatentSpaceSamples
from .train import LightningSyllableClassifier


class InceptionScorer(pl.LightningModule):
    """
    Lightning module used to compute Inception Score in a distributed fashion.

    Parameters
    ----------
        classifier : torch.Module
            Syllable classifier.
        generator : torch.Module
            GAN syllable generator.
        n_split : int, default to 10
            Number of dataset splits to evaluate Inception Score
            variance.
    """
    def __init__(self, classifier, generator=None, n_split=10):
        super().__init__()
        self.generator = generator
        self.classifier = classifier
        self.n_split = n_split
        self.inception = InceptionScore(feature=self.classifier, splits=n_split)
        self.is_mean = None
        self.is_std = None

    def forward(self, z):
        if self.generator is None:
            return z
        x = self.generator(z)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = self(batch)
        self.inception.update(x)
        return x

    def on_predict_end(self):
        self.is_mean, self.is_std = self.inception.compute()


def baseline_inception_score(
    scorer_ckpt,
    data_dir,
    out_dir,
    batch_size=100,
    n_split=10,
    devices=1,
    num_nodes=1,
    num_workers=12,
    seed=0,
    **kwargs,
):
    """
    Estimate baseline Inception Score (obtained from real data).
    
    Parameters
    ----------
    scorer_ckpt : str
        Path to classifier checkpoint.
    data_dir : str
        Syllable dataset directory.
    out_dir : str
        Output directory, where metrics will be saved.
    n_split : int, default to 10
        Number of dataset splits to evaluate Inception Score
        variance.
    batch_size : int, default to 100
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
    seed : int, default to 0
    """
    pl.seed_everything(seed, workers=True)

    data_dir = Path(data_dir)

    dataset = Canary16kDataset(dataset_path=data_dir, with_classes=False)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    scorer_ckpt = Path(scorer_ckpt)
    classifier = LightningSyllableClassifier.load_from_checkpoint(
        scorer_ckpt
    ).classifier

    scorer = InceptionScorer(classifier=classifier, n_split=n_split)

    engine = pl.Trainer(
        accelerator="auto",
        devices=devices,
        num_nodes=num_nodes,
        strategy="auto",
    )

    features = engine.predict(scorer, dataloader)

    is_mean = float(scorer.is_mean.cpu().numpy())
    is_std = float(scorer.is_std.cpu().numpy())

    scores = {}
    scores["epoch"] = "baseline"
    scores["is_mean"] = is_mean
    scores["is_std"] = is_std

    print(f"IS - baseline: {is_mean}±{is_std}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(f"{str(out_dir)}/is_scores-baseline.yml", "w+") as fp:
        yaml.dump(scores, fp, Dumper=yaml.Dumper)

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        str(pred_dir / "prediction_baseline.npy"),
        torch.concatenate(features, dim=0).cpu().numpy(),
    )


def inception_score(
    scorer_ckpt,
    generator_version,
    out_dir,
    vec_file=None,
    n_samples=50000,
    latent_dim=3,
    n_split=10,
    batch_size=100,
    devices=1,
    num_nodes=1,
    num_workers=12,
    seed=0,
    **kwargs,
):
    """
    Estimate syllable generator Inception Score.
    
    Parameters
    ----------
    scorer_ckpt : str
        Path to classifier checkpoint.
    generator_version : str
        GAN checkpoints directory.
    out_dir : str
        Output directory, where metrics will be saved.
    vec_file : str, optional
        Path to Numpy file storing latent vectors, for reproducibility.
        If unused, latent vectors will be randomly sampled at
        execution.
    n_samples : int, default to 50000
        Number of audio samples to generate for Inception Score estimation.
    latent_dim : int, default to 3
        GAN generator latent space dimension.
    n_split : int, default to 10
        Number of dataset splits to evaluate Inception Score
        variance.
    batch_size : int, default to 100
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
    seed : int, default to 0
    """
    pl.seed_everything(seed, workers=True)

    vec_file = Path(vec_file) if vec_file is not None else None

    dataset = LatentSpaceSamples(
        vec_file=vec_file, n_samples=n_samples, latent_dim=latent_dim
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    scorer_ckpt = Path(scorer_ckpt)
    classifier = LightningSyllableClassifier.load_from_checkpoint(
        scorer_ckpt
    ).classifier

    generator_ckpts = Path(generator_version) / "all"

    scores = {
        "epoch": [],
        "is_mean": [],
        "is_std": [],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for generator_ckpt in generator_ckpts.rglob("*.ckpt"):
        epoch = int(generator_ckpt.stem.split("-")[0].split("_")[-1])

        generator_ckpt = Path(generator_ckpt)
        generator = CanaryGAN.load_from_checkpoint(generator_ckpt).generator
        scorer = InceptionScorer(generator=generator, classifier=classifier, n_split=n_split)
        engine = pl.Trainer(
            accelerator="auto",
            devices=devices,
            num_nodes=num_nodes,
            strategy="auto",
        )

        features = engine.predict(scorer, dataloader)

        is_mean = float(scorer.is_mean.cpu().numpy())
        is_std = float(scorer.is_std.cpu().numpy())

        scores["epoch"].append(epoch)
        scores["is_mean"].append(is_mean)
        scores["is_std"].append(is_std)

        print(f"IS - epoch {epoch}: {is_mean}±{is_std}")


    with open(
        f"{str(out_dir)}/is_scores-{Path(generator_version).name}.yml", "w+"
    ) as fp:
        yaml.dump(scores, fp, Dumper=yaml.Dumper)
