# Author: Nathan Trouvain at 15/12/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from pathlib import Path
import dill as pickle

import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ...decoder import ESNDecoder, SVMDecoder, KNNDecoder, DecoderDataset


_decoder_map = {"esn": ESNDecoder, "svm": SVMDecoder, "knn": KNNDecoder}

_epilog = """Note: When using --reservoirs, make sure all reservoir filenames ends with "seed-[n].reservoir.pkl".
This will allow to select a specific reservoir using its corresponding seed, with the --seed parameter.

Note: When using --preprocessed_dir instead of --data_dir, all preprocessing parameters (--n_fft, --n-mfcc...) will be
ignored. The dataset and its parameters will be loaded from the checkpoint.
"""


@click.command(
    name="train-decoders", help="ESN, kNN and SVM decoders training.", epilog=_epilog
)
@click.option(
    "-s",
    "--save-dir",
    type=str,
    required=True,
    help="Model directory (checkpoints, logs...).",
)
@click.option(
    "-p",
    "--preprocessed-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Preprocessed dataset.",
)
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Dataset directory.",
)
@click.option(
    "--reservoirs",
    "instance-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing pickled reservoirpy Reservoir objects to use when creating ESN decoders.",
)
@click.option(
    "-m",
    "--model",
    "models",
    type=click.Choice(["esn", "svm", "knn"]),
    multiple=True,
    default=["esn", "svm", "knn"],
    help="Model to train. May be used several time to train multiple decoders at the same time.",
)
@click.option("--seed", type=int, help="Random seed.")
@click.option(
    "--dim",
    type=int,
    default=13,
    help="Input dimension of decoders. If using MFCC, must be the number of coefficients.",
)
@click.option(
    "--features",
    type=click.Choice(["deltas", "spec", "mel", "mfcc-classic"]),
    default="deltas",
    help="Type of preprocessing.",
)
@click.option(
    "--n-mfcc",
    type=int,
    default=13,
    help="Number of MFCC to compute, when using 'deltas' or 'mfcc-classic'.",
)
@click.option(
    "--split",
    type=float,
    default=0.1,
    help="Proportion of data used as testing material.",
)
@click.option(
    "--sampling_rate", type=int, default=16000, help="Audio sampling rate, in Hz."
)
@click.option(
    "--norm-audio",
    is_flag=True,
    help="If used, will normalize audio values prior to preprocessing.",
)
@click.option(
    "--max-length",
    type=float,
    default=1.0,
    help="Max length of audio samples, in second. Will pad or cut samples to reach that value.",
)
@click.option(
    "--mfcc/--no-mfcc",
    default=False,
    help="If used, will return MFCC. Else, will only return MFCC derivatives and second order derivatives.",
)
@click.option(
    "--lifter", type=int, default=0, help="Liftering parameter, to rescale MFCC."
)
@click.option(
    "--padding",
    type=str,
    default="interp",
    help="Padding mode for derivative computation.",
)
@click.option(
    "--hop-length",
    type=int,
    default=128,
    help="Spectrogram window strides, in timesteps.",
)
@click.option(
    "--win-length",
    type=int,
    default=256,
    help="Spectrogram, window length, in timesteps.",
)
@click.option(
    "--fmin", type=float, default=500, help="Lower frequency boundary for MFCC, in Hz."
)
@click.option(
    "--fmax", type=float, default=8000, help="Upper frequency boundary for MFCC, in Hz."
)
@click.option("--n-fft", type=int, default=512, help="Number of FFT bins computed.")
@click.option("--n-mels", type=int, default=128, help="Number of Mel bins.")
def train_decoders(
    save_dir,
    instance_dir,
    data_dir,
    preprocessed_dir,
    models,
    seed,
    dim,
    **kwargs,
):
    if data_dir is None and preprocessed_dir is None:
        raise FileNotFoundError(
            "No data! data_dir and preprocessed_dir can't be None at the same time."
        )

    if preprocessed_dir is not None:
        dataset = DecoderDataset.from_checkpoint(Path(preprocessed_dir))
    else:
        dataset = DecoderDataset(data_dir=data_dir, seed=seed, **kwargs).get()

    x_train, y_train = dataset.train_data
    x_test, y_test = dataset.test_data

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name in models:

        click.echo(f"Training {name}")

        decoder_cls = _decoder_map[name]

        if name == "esn":
            if instance_dir is not None:
                instance_dir = Path(instance_dir)
                instance = list(instance_dir.glob(f"*seed_{seed}.reservoir.pkl"))[0]
                with open(instance, "rb") as fp:
                    reservoir = pickle.load(fp)

                decoder = ESNDecoder.from_reservoir(
                    reservoir,
                    name=f"esn-{seed}",
                    ridge=1e-5,
                )
            else:
                decoder = ESNDecoder(dim)
        else:
            decoder = decoder_cls()

        decoder.fit(x_train, y_train)

        y_pred = decoder.predict(x_test)

        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        cm = confusion_matrix(y_test, y_pred)

        decoder.save(save_dir / f"{name}.ckpt.pkl")

        np.savez(
            save_dir / f"{'-'.join(['metrics', name])}.npz",
            confusion=cm,
            accuracy=accuracy,
        )

        labels = dataset.class_labels

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"Accuracy: {accuracy:.3f}")
        im = ax.imshow(cm, cmap="inferno", interpolation="none")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_xlabel("Prediction")
        ax.set_xlabel("Truth")
        fig.colorbar(im, ax=ax)
        fig.savefig(
            Path(save_dir) / f"{'-'.join(['confusion', name])}.pdf",
            bbox_inches="tight",
        )

        click.echo("Done.")

    return
