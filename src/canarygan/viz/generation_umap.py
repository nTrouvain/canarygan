import dill as pickle
from pathlib import Path

import librosa as lbr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tqdm.auto import tqdm
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import cdist

SR = 16000
MAX_LENGTH = round(SR * 0.3)
HOP_LENGTH = 128
WIN_LENGTH = 256
N_FFT = 1024


class AlreadyComputed(Exception):
    pass


def make_spectrogram(x):
    s = lbr.stft(x, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_fft=N_FFT)
    return lbr.power_to_db(np.abs(s) ** 2)


def fetch_dataset(data_dir):
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
    fields = ["UMAP 1", "UMAP 2", "hdbscan_y", "hdbscan_remap"]

    data_table = {f: gen_df[f].values for f in fields}

    gen_dir = Path(gen_dir)
    file = list(gen_dir.rglob(f"version_{version}/*epoch_{str(epoch).zfill(3)}*.npz"))[
        0
    ]
    d = np.load(file)
    np.savez(file, **d, **data_table)


def save_umap_dataset(real_df, data_dir):
    fields = ["file", "label", "UMAP 1", "UMAP 2", "hdbscan_y", "hdbscan_remap"]
    selected = real_df[fields]

    file = Path(data_dir) / "umap_table.csv"

    selected.to_csv(file, index=False)


def create_reducer(real_df):

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
    clusterer = HDBSCAN(
        min_cluster_size=500, min_samples=250, n_jobs=-1, cluster_selection_method="eom"
    )
    return clusterer.fit_predict(real_df[["UMAP 1", "UMAP 2"]])


def concatenate_spectrograms(df):
    return np.concatenate(np.dstack(df.spec.values)).T


def embed(reducer, df):
    embeddings = reducer.transform(concatenate_spectrograms(df))
    df[["UMAP 1", "UMAP 2"]] = embeddings
    return df


def label_from_cluster(mixed_df):

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


def plot_mean_spectrograms(save_file, df, label_col="hdbscan_y"):

    labels = sorted(df[label_col].unique())

    n = len(labels)
    n_cols = 4
    n_rows = n // 4 + int(n % 4 > 0)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, lbl in enumerate(labels):
        ax = axs[i]
        q = df.query(f"{label_col}==@lbl")
        ax.imshow(
            np.mean(q.spec.values, axis=0),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="magma",
        )
        ax.set_title(f"{lbl} (n={len(q)})")

    fig.savefig(save_file, bbox_inches="tight")
    return


def plot_generation_umap(
    data_dir,
    gen_dir,
    version,
    epoch,
    save_file,
    reducer_ckpt=None,
    redo=False,
):

    save_file = Path(save_file)

    gen_df = fetch_generation(gen_dir, version, epoch, redo)
    real_df = fetch_dataset(data_dir)

    mixed_df = pd.concat([real_df, gen_df]).reset_index(drop=True)

    if reducer_ckpt is not None and Path(reducer_ckpt).is_file():
        with open(reducer_ckpt, "rb") as fp:
            reducer = pickle.load(fp)
    else:
        # Somehow only this works: fit on real data and embed both real/fake
        # at the same time
        reducer = create_reducer(real_df)

        if reducer_ckpt is not None:
            with open(reducer_ckpt, "wb+") as fp:
                pickle.dump(reducer, fp)

    mixed_df = embed(reducer, mixed_df)

    mixed_df["hdbscan_y"] = create_clusters(mixed_df)
    mixed_df = label_from_cluster(mixed_df)

    real_df = mixed_df.query("kind=='real'")
    gen_df = mixed_df.query("kind=='gen'")

    save_umap_generation(gen_df, gen_dir, version, epoch)

    if not (Path(data_dir) / "umap_table.csv").is_file() and not redo:
        save_umap_dataset(real_df, data_dir)

    save_specs = save_file.parent / ("mean_specs-" + save_file.name)
    plot_mean_spectrograms(save_specs, gen_df, label_col="hdbscan_y")

    save_specs = save_file.parent / ("remaped_mean_specs-" + save_file.name)
    plot_mean_spectrograms(
        save_specs, gen_df.query("hdbscan_remap!='X'"), label_col="hdbscan_remap"
    )

    # GAN quality plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    g = sns.scatterplot(
        data=gen_df,
        x="UMAP 1",
        y="UMAP 2",
        hue="label",
        s=50,
        alpha=0.5,
        marker="x",
        linewidths=2,
        palette="dark",
        ax=ax,
    )
    _ = sns.kdeplot(
        data=real_df,
        x="UMAP 1",
        y="UMAP 2",
        hue="label",
        palette="tab20",
        alpha=0.5,
        levels=5,
        zorder=0,
        legend=False,
        ax=ax,
    )

    labels = real_df.label.unique()
    for i, lbl in enumerate(labels):
        df_l = real_df.query("label==@lbl")
        top = df_l["UMAP 2"].median()
        mean = df_l["UMAP 1"].median()
        ax.text(mean, top, lbl, c=sns.color_palette("tab20")[i], size=12, weight="bold")

    g.legend(markerscale=1)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    fig.savefig(save_file, bbox_inches="tight")
    plt.close()

    # Decoder quality plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    g = sns.scatterplot(
        data=gen_df.sort_values(by="y").query("y!='X'"),
        x="UMAP 1",
        y="UMAP 2",
        hue="y",
        s=5,
        alpha=0.5,
        ec=None,
        linewidths=2,
        palette="tab20",
        ax=ax,
    )
    _ = sns.kdeplot(
        data=real_df,
        x="UMAP 1",
        y="UMAP 2",
        hue="label",
        palette="tab20",
        alpha=0.5,
        levels=5,
        zorder=0,
        legend=False,
        ax=ax,
    )

    labels = real_df.label.unique()
    for i, lbl in enumerate(labels):
        df_l = real_df.query("label==@lbl")
        top = df_l["UMAP 2"].median() + 1
        mean = df_l["UMAP 1"].median() - 4
        ax.text(mean, top, lbl, c=sns.color_palette("tab20")[i], size=12, weight="bold")

    g.legend(markerscale=5)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    save_file_2 = save_file.parent / ("decoded-" + save_file.name)
    fig.savefig(save_file_2, bbox_inches="tight")
    plt.close()

    # HDBSCAN plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    g = sns.scatterplot(
        data=gen_df,
        x="UMAP 1",
        y="UMAP 2",
        hue="hdbscan_y",
        s=5,
        alpha=0.5,
        ec=None,
        linewidths=2,
        palette="tab20",
        ax=ax,
    )
    _ = sns.kdeplot(
        data=real_df,
        x="UMAP 1",
        y="UMAP 2",
        hue="label",
        palette="tab20",
        alpha=0.5,
        levels=5,
        zorder=0,
        legend=False,
        ax=ax,
    )

    labels = real_df.label.unique()
    for i, lbl in enumerate(labels):
        df_l = real_df.query("label==@lbl")
        top = df_l["UMAP 2"].median() + 1
        mean = df_l["UMAP 1"].median() - 4
        ax.text(mean, top, lbl, c=sns.color_palette("tab20")[i], size=12, weight="bold")

    g.legend(markerscale=5)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    save_file_ = save_file.parent / ("hdbscan-" + save_file.name)
    fig.savefig(save_file_, bbox_inches="tight")
    plt.close()

    # Remapped HDBSCAN labels
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    g = sns.scatterplot(
        data=gen_df.sort_values(by="hdbscan_remap"),
        x="UMAP 1",
        y="UMAP 2",
        hue="hdbscan_remap",
        s=5,
        alpha=0.5,
        ec=None,
        linewidths=2,
        palette="tab20",
        ax=ax,
    )
    _ = sns.kdeplot(
        data=real_df,
        x="UMAP 1",
        y="UMAP 2",
        hue="label",
        palette="tab20",
        alpha=0.5,
        levels=5,
        zorder=0,
        legend=False,
        ax=ax,
    )

    labels = real_df.label.unique()
    for i, lbl in enumerate(labels):
        df_l = real_df.query("label==@lbl")
        top = df_l["UMAP 2"].median() + 1
        mean = df_l["UMAP 1"].median() - 4
        ax.text(mean, top, lbl, c=sns.color_palette("tab20")[i], size=12, weight="bold")

    g.legend(markerscale=5)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    save_file_ = save_file.parent / ("remaped_hdbscan" + save_file.name)
    fig.savefig(save_file_, bbox_inches="tight")
    plt.close()

    return
