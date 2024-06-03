# Author: Nathan Trouvain at 10/20/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Utilities to manipulate and transform canary syllable data for
decoding.
"""
import yaml

from pathlib import Path

import numpy as np
import pandas as pd

from rich.progress import track
from sklearn.model_selection import train_test_split

from .preprocess import preprocess


def as_path(path):
    return Path(path) if path is not None else path


def get_spec(df, feat="feat"):
    """Concatenate spectrograms stored in a Pandas DataFrame,
    in column "feat". """
    return np.vstack([s[np.newaxis, ...] for s in df[feat].to_numpy()])


class DecoderDataset:
    """
    Utility class to load, preprocess and save datasets
    of canary syllable sounds.

    Data must be provided as WAV files in subdirectories (relative to
    data_dir) named after their syllable class.

    Parameters
    ----------
    data_dir : str, optional
        Root dataset directory.
    checkpoint_dir : str, optional
        Directory name for dataset transformed content and parameters.
    split : float, default to 0.1
        Proportion of data used as testing material.
    sampling_rate : int, default to 16000,
        Audio sampling rate, in Hz.
    features : ["deltas", "spec", "mel", "mfcc-classic"], default to deltas
        Type of features to extract:
            - deltas: MFCC and derivatives
            - spec: spectrogam
            - mel: mel-spectrogam
            - mfcc-classic: MFCC and derivatives as done in Pagliarini et al. (2018).
    norm_audio : bool, default to False
        If True, will normalize audio values prior to preprocessing.
    max_length : float, default to 1.0
        Max length of audio samples, in second. Will pad or cut samples
        to reach that value.
    mfcc : bool, default to False
        If True, will return MFCC. If False, will only return MFCC
        derivatives and second order derivatives.
    lifter : int, default to 0
        Liftering parameters, to rescale MFCC.
    n_mfcc : int, default to 13
        Number of coefficients to compute.
    padding : str, default to "interp"
        Padding mode for derivative computation.
    hop_length : int, default to 128
        Spectrogram window strides, in timesteps.
    win_length : int, default to 256
        Spectrogram window length, in timesteps.
    fmin : float, default to 500
        Lower frequency boundary, in Hz.
    fmax : float, default to 8000
        Upper frequency boundary, in Hz.
    n_fft : int, default to 512
        Number of FFT bins computed.
    n_mels : int, default to 128
        Number of Mel bins.
    seed: int, default to 0
        Random state seed.
    """
    @classmethod
    def from_checkpoint(cls, checkpoint_dir):
        """
        Load dataset from a checkpoint directory, containing
        saved transformed data and preprocessing parameters.
        """
        dataset = cls()

        ckpt_dir = Path(checkpoint_dir)

        if (checkpoint_dir / "decoder-dataset.config.yml").exists():
            dataset.load_params(checkpoint_dir / "decoder-dataset.config.yml")

        if not (ckpt_dir / "decoder-dataset.mfcc.npz").is_file():
            train_data = test_data = None
        else:
            data = np.load(ckpt_dir / "decoder-dataset.mfcc.npz")

            train_data = data["x_train"]
            test_data = data["x_test"]
            train_labels = data["y_train"]
            test_labels = data["y_test"]

            train_data = (train_data, train_labels)
            test_data = (test_data, test_labels)

            dataset.train_data = train_data
            dataset.test_data = test_data

        return dataset

    def __init__(
        self,
        data_dir=None,
        checkpoint_dir=None,
        split=0.1,
        features="deltas",
        sampling_rate=16000,
        max_length=1.0,
        norm_audio=False,
        n_mfcc=13,
        n_mels=128,
        n_fft=512,
        mfcc=False,
        fmin=500,
        fmax=8000,
        hop_length=128,
        win_length=256,
        lifter=0,
        padding="interp",
        seed=0,
    ):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.train_data = None
        self.test_data = None

        self.split = split
        self.seed = seed

        self.transform_kwargs = dict(
            features=features,
            sampling_rate=sampling_rate,
            max_length=max_length,
            norm_audio=norm_audio,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            win_length=win_length,
            lifter=lifter,
            padding=padding,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=n_fft,
            mfcc=mfcc,
        )

        self.class_labels = None
        self.class_to_idx = None
        self.idx_to_class = None

    def __len__(self):
        return sum([len(v) for v in self.data_files.values()])

    def save_params(self, to_file):
        """
        Save dataset and preprocessing parameters to a YAML file.
        """
        save_file = Path(to_file)

        heavy_data = ["train_data", "test_data"]

        params = {}
        for k, v in self.__dict__.items():
            if k not in heavy_data:
                if isinstance(v, Path):
                    params[k] = str(v)
                else:
                    params[k] = v

        with save_file.open("w+") as fp:
            yaml.dump(params, fp, Dumper=yaml.Dumper)

        return self

    def load_params(self, from_file):
        """
        Load a YAML file containing dataset and preprocessing
        parameters.
        """
        with Path(from_file).open("r") as fp:
            params = yaml.load(fp, Loader=yaml.Loader)
        for k, v in params.items():
            self.__setattr__(k, v)
        return self

    def get(self, data_dir=None, reload=False):
        """
        Load and preprocess data.

        If data was already loaded and transformed, 
        no operation will be applied (unless reload is set to True).
        
        This method will populate the "train_data" and "test_data"
        attributes.

        Parameters
        ----------
        data_dir : str, optional
            Path to dataset directory, if not already given.
        reload : bool, default to False
            If True, will perform preprocessing even if already
            performed.

        Returns
        -------
        Dataset
            Self.
        """
        if self.train_data is not None and self.test_data is not None and not reload:
            return self

        if self.data_dir is None:
            if data_dir is None:
                raise ValueError(
                    "data_dir is None and DecoderDataset.data_dir "
                    "is None. Don't know from where to get data."
                )
            else:
                self.data_dir = data_dir
                data_dir = as_path(data_dir)
        else:
            data_dir = as_path(self.data_dir)

        files = list(Path(data_dir).rglob("**/*.wav"))

        data_table = {
            "file": files,
            "label": [str(f.parent.name) for f in files],
        }

        df = pd.DataFrame(data_table)

        df["label"] = df["label"].astype(str)

        df = df.sort_values(by=["label", "file"]).reset_index(drop=True)

        # Class labels are directory names in the root dataset directory.
        self.class_labels = sorted(df.label.unique().tolist())

        print(f"Found {len(self.class_labels)} classes.")
        print(f"{self.class_labels}")

        self.class_to_idx = {c: i for i, c in enumerate(self.class_labels)}
        self.idx_to_class = {i: c for i, c in enumerate(self.class_labels)}

        df["feat"] = preprocess(track(df["file"]), **self.transform_kwargs)

        train_df, test_df = train_test_split(
            df,
            test_size=self.split,
            stratify=df["label"],
            random_state=self.seed,
        )

        self.train_data = (
            get_spec(train_df, feat="feat"),
            train_df["label"].to_numpy().astype(str),
        )

        self.test_data = (
            get_spec(test_df, feat="feat"),
            test_df["label"].to_numpy().astype(str),
        )

        return self

    def checkpoint(self, checkpoint_dir=None):
        """
        Save the preprocessed data and preprocessing parameters
        to disk.
        """
        if self.checkpoint_dir is None:
            if checkpoint_dir is None:
                raise ValueError(
                    "checkpoint_dir is None and DecoderDataset.checkpoint_dir "
                    "is None. Don't know where to checkpoint."
                )
            else:
                ckpt_dir = as_path(checkpoint_dir)
                self.checkpoint_dir = checkpoint_dir
        else:
            ckpt_dir = as_path(self.checkpoint_dir)

        train_data, train_labels = self.train_data
        test_data, test_labels = self.test_data

        if self.train_data is None:
            raise ValueError("Can't checkpoint: no data")

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            ckpt_dir / "decoder-dataset.mfcc.npz",
            x_train=train_data,
            y_train=train_labels,
            x_test=test_data,
            y_test=test_labels,
        )

        self.save_params(ckpt_dir / "decoder-dataset.config.yml")

        return self

