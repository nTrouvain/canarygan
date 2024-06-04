# Author: Nathan Trouvain at 15/12/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from pathlib import Path
import argparse
import dill as pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from pmseq.decoder.base import ESNDecoder, SVMDecoder, KNNDecoder
from pmseq.decoder.dataset import DecoderDataset


parser = argparse.ArgumentParser(description="Decoders training.")

parser.add_argument(
    "--save_dir", type=str, help="Model directory (checkpoints, logs...)."
)
parser.add_argument("--instance_dir", type=str)
parser.add_argument("--preprocessed_dir", type=str, help="Preprocessed dataset.")
parser.add_argument("--seed", type=int, help="Random seed.")
parser.add_argument("--dim", type=int, default=13)

SAMPLING_RATE = 16000
# MAX_TIME = round(0.3 * SAMPLING_RATE)

if __name__ == "__main__":
    args = parser.parse_args()

    seed = args.seed

    dataset = DecoderDataset.from_checkpoint(Path(args.preprocessed_dir))

    x_train, y_train = dataset.train_data
    x_test, y_test = dataset.test_data

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.instance_dir is not None:
        instance_dir = Path(args.instance_dir)
        instance = list(instance_dir.glob(f"*seed_{seed}.reservoir.pkl"))[0]
        with open(instance, "rb") as fp:
            reservoir = pickle.load(fp)

        esn = ESNDecoder.from_reservoir(
            reservoir,
            name=f"esn-{seed}",
            ridge=1e-5,
        )
    else:
        esn = ESNDecoder(dim=args.dim)

    decoders = dict(esn=esn, knn=KNNDecoder(), svm=SVMDecoder())

    for name, decoder in decoders.items():

        if name == "esn":
            decoder.fit(x_train, y_train)
        else:
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
