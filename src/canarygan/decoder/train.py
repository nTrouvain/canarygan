# Author: Nathan Trouvain at 10/20/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import random
import dill as pickle

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import softmax
from sklearn.metrics import confusion_matrix

from .dataset import DecoderDataset
from .model import esn


def load_dataset(data_dir=None, ckpt_dir=None, reload=False):
    if data_dir is None and ckpt_dir is None:
        raise ValueError("data_dir and/or ckpt_dir must be given.")

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else None
    data_dir = Path(data_dir) if data_dir is not None else None

    if (
        ckpt_dir is not None
        and (ckpt_dir / "decoder-dataset.mfcc.npz").exists()
        and not reload
    ):
        dataset = DecoderDataset.from_checkpoint(ckpt_dir)
    else:
        if ckpt_dir is None:
            ckpt_dir = Path("data/decoder")
        dataset = DecoderDataset(data_dir).get(reload=reload).checkpoint(ckpt_dir)

    return dataset


def train(
    save_dir,
    data_dir=None,
    preprocessed_dir=None,
    reload=False,
    split=True,
    seed=None,
):
    random.seed(seed)
    np.random.seed(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_dir = save_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir = save_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_dir, preprocessed_dir, reload)
    model = esn(name=str(seed))

    train_data, train_labels = dataset.train_data

    x_train = [np.concatenate(x[1:], axis=1) for x in train_data]

    if not split:
        test_data, _ = dataset.test_data
        x_test = [np.concatenate(x[1:], axis=1) for x in test_data]
        x_train += x_test

    n = len(x_train)
    t = x_train[0].shape[0]

    # One-hot encode
    idx_map = dataset.class_to_idx
    train_labels = np.array([idx_map[y[0, 0]] for y in train_labels])
    y_train = np.zeros((n, t, train_labels.max() + 1))
    y_train[np.arange(n), :, train_labels] = 1

    model = model.fit(x_train, y_train)

    model.class_labels = dataset.class_labels
    model.class_to_idx = dataset.class_to_idx

    with (model_dir / f"decoder-{seed}.pkl").open("wb+") as fp:
        pickle.dump(model, fp)

    # Test
    if split:
        test(model, dataset, report_dir, name=str(seed))

    return


def test(
    model,
    dataset,
    report_dir,
    name="",
):
    test_data, test_labels = dataset.test_data
    x_test = [np.concatenate(x[1:], axis=1) for x in test_data]

    idx_map = dataset.class_to_idx
    y_test = np.array([idx_map[y[0, 0]] for y in test_labels])

    y_pred = model.run(x_test)

    if isinstance(y_pred, list):
        y_pred = np.r_[y_pred]

    y_pred = np.argmax(softmax(np.sum(y_pred, axis=1), axis=1), axis=1)

    accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
    cm = confusion_matrix(y_test, y_pred)

    np.savez(
        Path(report_dir) / f"{'-'.join(['metrics', name])}.npz",
        confusion=cm,
        accuracy=accuracy,
    )

    labels = dataset.class_labels

    fig, ax = plt.subplots()
    ax.set_title(f"Accuracy: {accuracy:.3f}")
    im = ax.imshow(cm, cmap="inferno", interpolation="none")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_xlabel("Prediction")
    ax.set_xlabel("Truth")
    fig.colorbar(im, ax=ax)
    fig.savefig(
        Path(report_dir) / f"{'-'.join(['confusion', name])}.pdf", bbox_inches="tight"
    )

    return


if __name__ == "__main__":
    train(
        model_dir="./models/decoder/",
        ckpt_dir="./data/decoder-preprocessed",
    )
