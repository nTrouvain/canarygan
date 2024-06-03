# Author: Nathan Trouvain <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
UMAP and clustering tools for figures.
"""
from pathlib import Path

import librosa as lbr
import numpy as np
import pandas as pd
import umap
from tqdm.auto import tqdm
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import cdist

# Sampling rate
SR = 16000
# Max length of audio samples
MAX_LENGTH = round(SR * 0.3)
# FFT stride
HOP_LENGTH = 128
# FFT window size
WIN_LENGTH = 256
# Number of FFT bins
N_FFT = 1024


class AlreadyComputed(Exception):
    pass


def make_spectrogram(x):
    s = lbr.stft(x, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_fft=N_FFT)
    return lbr.power_to_db(np.abs(s) ** 2)


def concatenate_spectrograms(df):
    return np.concatenate(np.dstack(df.spec.values)).T


def fetch_dataset(data_dir):
    """
    Load dataset of real canary syllables.
    Compute spectrograms, and store everything in a DataFrame.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.rglob("**/*.wav"))

    data_table = {
        "file": files,
        "label": [f.parent.name for f in files],
        "x": [lbr.load(f, sr=SR)[0][:MAX_LENGTH] for f in tqdm(files, "Loading real")],
        "sr": SR,
        "kind": "real",
    }
    df = pd.DataFrame(data_table).sort_values(by=["label", "file"])
    df["y"] = df["label"]
    df["spec"] = list(make_spectrogram(np.vstack(df.x.values)))
    return df


def fetch_generation(gen_dir, version, epoch, redo=False):
    """
    Load dataset of generated canary syllables.
    Compute spectrograms, and store everything in a DataFrame.
    """
    gen_dir = Path(gen_dir)
    file = list(gen_dir.rglob(f"version_{version}/*epoch_{str(epoch).zfill(3)}*.npz"))[
        0
    ]
    d = np.load(file)

    if np.any(["UMAP" in k for k in d.keys()]) and not redo:
        raise AlreadyComputed()

    data_table = {
        "file": file,
        "version": int(d["version"]),
        "epoch": int(d["epoch"]),
        "y": d["y_argmax"].tolist(),
        "sr": SR,
        "z": list(d["z"]),
        "x": list(d["x"][:, :MAX_LENGTH]),
        "kind": "gen",
    }

    gdf = pd.DataFrame(data_table)
    
    # Transform epoch labels for easier visualisation.
    def get_epoch_label(e):
        if e < 300:
            return "EARLY"
        elif 300 <= e <= 700:
            return "MID"
        else:
            return "LATE"

    # gdf = gdf.explode(["x", "y", "z"]).reset_index(drop=True)
    gdf["spec"] = list(make_spectrogram(np.vstack(gdf.x.values)))

    gdf["y"] = gdf["y"].replace(
        {k: "X" for k in gdf["y"].unique() if "EARLY" in k or "OT" in k or "WN" in k}
    )

    gdf["label"] = gdf.epoch.replace(
        {e: f"{get_epoch_label(e)}_{e}" for e in gdf.epoch.values}
    )

    return gdf


def save_umap_generation(gen_df, gen_dir, version, epoch):
    """
    Save UMAP embeddings and HDBSCAN clustering for each generated
    syllable sound.
    """
    fields = ["UMAP 1", "UMAP 2", "hdbscan_y", "hdbscan_remap"]

    data_table = {f: gen_df[f].values for f in fields}

    gen_dir = Path(gen_dir)
    file = list(gen_dir.rglob(f"version_{version}/*epoch_{str(epoch).zfill(3)}*.npz"))[
        0
    ]
    d = np.load(file)
    np.savez(file, **d, **data_table)


def save_umap_dataset(real_df, data_dir):
    """
    Save UMAP embeddings and HDBSCAN clustering for each
    real syllable sound.
    """
    fields = ["file", "label", "UMAP 1", "UMAP 2", "hdbscan_y", "hdbscan_remap"]
    selected = real_df[fields]

    file = Path(data_dir) / "umap_table.csv"

    selected.to_csv(file, index=False)


def create_reducer(real_df):
    """
    Fit UMAP on real syllable spectrograms.
    """
    reducer = umap.UMAP(
        n_neighbors=100,
        min_dist=0.9,
        n_components=2,
        metric="euclidean",
        verbose=True,
        random_state=42,
    )
    return reducer.fit(concatenate_spectrograms(real_df))


def create_clusters(real_df):
    """
    Fit HDBSCAN on real syllable UMAP projections.
    """
    clusterer = HDBSCAN(
        min_cluster_size=500, min_samples=250, n_jobs=-1, cluster_selection_method="eom"
    )
    return clusterer.fit_predict(real_df[["UMAP 1", "UMAP 2"]])


def embed(reducer, df):
    """
    Project spectrograms into UMAP space.
    """
    embeddings = reducer.transform(concatenate_spectrograms(df))
    df[["UMAP 1", "UMAP 2"]] = embeddings
    return df


def label_from_cluster(mixed_df):
    """
    Assign label to syllables based on HDBSCAN clusters.
    For each generated syllable within an HDBSCAN cluster (not being assigned
    the -1 class), find the closest real syllable in UMAP space and
    copy its class label.
    HDBSCAN out-of-distribution class "-1" is remaped to "X".
    """
    real_df = mixed_df.query("kind=='real'")
    gen_df = mixed_df.query("kind=='gen'")

    real_emb = real_df[["UMAP 1", "UMAP 2"]].values
    gen_emb = gen_df[["UMAP 1", "UMAP 2"]].values

    closest_real = np.argmin(cdist(gen_emb, real_emb), axis=1)
    closest_real_label = real_df.iloc[closest_real]["label"].values

    mixed_df["hdbscan_remap"] = mixed_df["y"]
    mixed_df.loc[gen_df.index, "hdbscan_remap"] = closest_real_label
    # Out of cluster samples for HDBSCAN become X class
    mixed_df.loc[gen_df.query("hdbscan_y==-1").index, "hdbscan_remap"] = "X"

    return mixed_df
