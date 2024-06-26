# Author: Nathan Trouvain at 8/30/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Dataset related operations for CanaryGAN.
"""
import pathlib

import torch
import torchaudio
import lightning as pl
import numpy as np

from torch.utils import data
from sklearn.model_selection import train_test_split

from .const import SliceLengths


class DummyNoise(data.Dataset):
    """
    Just some random signal, for testing.
    """

    def __init__(self):
        self.data = torch.rand(16000, 1, SliceLengths.SHORT.value)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        return self.data[item]


class LatentSpaceSamples(data.Dataset):
    """
    Generate or load latent vectors for CanaryGAN.

    Parameters
    ----------
    vec_file : str, optional
        Path to a Numpy archive containing latent vectors.
        If None, random vectors will be generated at runtime.
    latent_dim : int, default to 3
        GAN latent space dimension.
    n_samples : int, default to 50000
        Number of latent vectors to generate if vec_file is None.
    """

    def __init__(self, vec_file=None, latent_dim=3, n_samples=50000):

        if vec_file is not None:
            self.latent_vects = torch.Tensor(np.load(vec_file))

            self.n_samples = self.latent_vects.size(0)
            self.latent_dim = self.latent_vects.size(1)
        else:
            self.n_samples = n_samples
            self.latent_dim = latent_dim

            # z ~ U[-1, 1]
            self.latent_vects = torch.rand(n_samples, latent_dim) * 2 - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.latent_vects[item]


class Canary16kDataset(data.Dataset):
    """
    Load and batch real canary syllables.

    Dataset must be provided as directories containing
    WAV files, where each directory name is a syllable class name.

    Parameters
    ----------
    dataset_path : str
        Path to dataset root directory.
    with_classes : bool, default to False
        If True, will yield class labels alongside audios
        when iterated (classifier training set).
        If False, only provide audio samples
        (GAN training set).
    """

    def __init__(self, dataset_path, with_classes=False):
        self.with_classes = with_classes
        self.dataset_path = dataset_path

        dataset_path = pathlib.Path(self.dataset_path)

        # Class labels are directory names in the root dataset directory.
        self.class_labels = sorted(
            [str(d.stem) for d in dataset_path.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_labels)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_labels)}

        blocks = []
        label_blocks = []
        for sylldir in dataset_path.iterdir():
            label = str(sylldir.stem)
            # tranform label to numerical index
            label_id = self.class_to_idx[label]

            files = sorted(sylldir.glob("*.wav"))

            label_block = torch.zeros(len(files))
            block = torch.zeros(len(files), 1, SliceLengths.SHORT.value)

            for i, file in enumerate(files):
                wave, _ = torchaudio.load(file)
                block[i, :, : wave.size(1)] = wave
                label_block[i] = label_id

            blocks.append(block)
            label_blocks.append(label_block)

        self.data = torch.concatenate(blocks, dim=0)
        self.labels = torch.concatenate(label_blocks, dim=0).to(torch.int64)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        if self.with_classes:
            return self.data[item], self.labels[item]
        return self.data[item]


class Canary16kDataModule(pl.LightningDataModule):
    """
    Utility class to provide Canary16kDataset as a
    Lightning compatible object for training the GAN.

    Parameters
    ----------
    dataset_path : str
        Path to dataset root directory.
    batch_size : int, default to 1
        Batch size for training/inference.
    split : bool, default to False
        If True, will split data into training and test sets.
    with_classes : bool, default to False
        If True, will yield class labels alongside audios
        when iterated (classifier training set).
        If False, only provide audio samples
        (GAN training set).
    num_workers : int, optional
        Number of processes used to load data.
    """

    def __init__(
        self,
        dataset_path,
        batch_size=1,
        split=False,
        with_classes=False,
        num_workers=None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.with_classes = with_classes
        self.num_workers = num_workers
        self.split = split

        self.test_set = None
        self.val_set = None
        self.train_set = None
        self.predict_set = None

    def setup(self, stage: str):
        dataset = Canary16kDataset(self.dataset_path, with_classes=self.with_classes)

        if self.split:
            all_train_idx, test_idx = train_test_split(
                np.arange(len(dataset)),
                test_size=0.1,  # 100 per class
                shuffle=True,
                random_state=4862,
                stratify=dataset.labels.detach().numpy(),
            )
            train_idx, val_idx = train_test_split(
                all_train_idx,
                test_size=0.05,  # 50 per class
                shuffle=True,
                random_state=4862,
                stratify=dataset.labels[all_train_idx].detach().numpy(),
            )
            self.train_set = data.Subset(dataset, train_idx)
            self.val_set = data.Subset(dataset, val_idx)
            self.test_set = data.Subset(dataset, test_idx)
        else:
            self.train_set = dataset
            self.val_set = dataset
            self.test_set = dataset

        self.predict_set = dataset

    def train_dataloader(self):
        self.train_set.with_classes = True
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        self.val_set.with_classes = True
        return data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        self.test_set.with_classes = True
        return data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        self.predict_set.with_classes = False
        return data.DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )
