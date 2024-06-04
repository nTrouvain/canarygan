# Author: Nathan Trouvain <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
GAN quality figures.
"""
import dill as pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .umap import fetch_dataset, fetch_generation
from .umap import create_clusters, create_reducer
from .umap import embed, label_from_cluster
from .umap import save_umap_dataset, save_umap_generation


def plot_mean_spectrograms(save_file, df, label_col="hdbscan_y"):
    """
    Plot mean spectrograms for each syllable label in dataset.

    Parameters
    ----------
    save_file : str
        File name for figure.
    df : DataFrame
        Dataframe containing all syllables metadata and spectrograms.
    label_col : str, default to "hdbscan_y"
        Name of the column in df storing the syllable labels.
    """
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
    """
    Make many plots displaying UMAP projections of real and generated syllables.
    Will also save UMAP projections and cluster labels in generated syllables metadata.

    Output plots:
        - Mean spectrograms per class of syllables (generated)
          with and without "X" class,
        - UMAP plot of real data (kernel density plot)
          against generated data (scatter plot) colored by decoder label,
        - Same, with HDBSCAN arbitrary labels,
        - Same, with HDBSCAN clusters remapped to real labels.

    Parameters
    ----------
    data_dir : str
        Real data directory.
    gen_dir : str
        Generated data directory.
    version : int
        GAN version to consider in gen_dir.
    epoch : int
        GAN training epoch to consider in gen_dir.
    save_file : str
        Base filename for figures.
    reducer_ckpt : str, optional
        Path to a trained and saved UMAP object.
    redo : bool, default to False
        If True, will redo UMAP projections even
        if they are already present in gen_dir.
    """

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
