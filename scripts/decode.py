import dill as pickle
from pathlib import Path
import argparse

import numpy as np
import reservoirpy as rpy

from rich.progress import track

from pmseq.decoder.preprocessing import preprocess


MAX_TIME = round(0.3 * 16000)


parser = argparse.ArgumentParser()
parser.add_argument("gen_file", type=str)
parser.add_argument("--umap_ckpt", type=str)
parser.add_argument("decoder_ckpt", type=str)
parser.add_argument("--features", type=str)
parser.add_argument("--n_mfcc", type=int, default=13)

preprocessing_args = dict(
    sampling_rate=16000,
    max_length=1.0,
    norm_audio=False,
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
)


if __name__ == "__main__":
    # args = parser.parse_args(
    #     [
    #         "/run/media/nathan/Nathan4T/Nathan-canarygan/generation/version_1/generation-version_1-epoch_989-step_89100.npz",
    #         "./reports/checkpoints",
    #         "--features",
    #         "mfcc-classic",
    #     ]
    # )

    args = parser.parse_args()

    rpy.verbosity(0)

    decoders = {}
    for path in Path(args.decoder_dir).glob("*.ckpt.pkl"):
        with path.open("rb") as fp:
            decoder = pickle.load(fp)

        if "esn" in path.name:
            decoders["esn"] = decoder
        elif "svm" in path.name:
            decoders["svm"] = decoder
        elif "knn" in path.name:
            decoders["knn"] = decoder
        else:
            raise ValueError(f"Unknown decoder: {path.name}")

    d = np.load(args.gen_file, allow_pickle=True)
    x = d["x"]

    x = preprocess(
        track(x), features=args.features, n_mfcc=args.n_mfcc, **preprocessing_args
    )
    x = np.vstack([s[np.newaxis, ...] for s in x])

    predictions = {}
    for name, decoder in track(decoders.items()):
        if f"y_{name}_{args.features}" in d:
            continue
        predictions[f"y_{name}_{args.features}"] = decoder.predict(x)

    if (umap_ckpt := args.umap_ckpt) is not None:
        umap_ckpt = Path(umap_ckpt)
        with umap_ckpt.open("rb") as fp:
            umap = pickle.load(fp)

        ...

    if len(predictions) != 0:
        np.savez_compressed(
            args.gen_file,
            **d,
            **predictions,
        )
