# Author: Nathan Trouvain at 10/20/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
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
    return np.vstack([s[np.newaxis, ...] for s in df[feat].to_numpy()])


class DecoderDataset:

    @classmethod
    def from_checkpoint(cls, checkpoint_dir):

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
        with Path(from_file).open("r") as fp:
            params = yaml.load(fp, Loader=yaml.Loader)
        for k, v in params.items():
            self.__setattr__(k, v)
        return self

    def get(self, data_dir=None, reload=False):
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

