import argparse
import time
import json
import uuid
from pathlib import Path
from os import PathLike

import yaml
import librosa as lbr
import numpy as np
import pandas as pd
import scipy.signal
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from rich.progress import track
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


HOP_LENGTH = 128
WIN_LENGTH = 256
N_FFT = 1024
SR = 16000
MAX_LENGTH = round(0.3 * SR)
N_MELS = 128
FMIN = 500
FMAX = 8000
N_MFCC = 13
REF_POW = 1.0
MIN_POW = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--gen_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--version", type=int, default=16)
parser.add_argument("--epoch", type=int, default=989)
parser.add_argument("--features", type=str, choices=["spec", "mel", "deltas", "base"])
parser.add_argument("--norm_audio", action="store_true")
parser.add_argument("--norm_spec", action="store_true")
parser.add_argument("--hop_length", type=float, default=HOP_LENGTH)
parser.add_argument("--win_length", type=float, default=WIN_LENGTH)
parser.add_argument("--n_fft", type=float, default=N_FFT)
parser.add_argument("--mfcc", action="store_true")


def make_spectrogram(
    x,
    ref=REF_POW,
    amin=MIN_POW,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_fft=N_FFT,
    **kwargs,
):
    _, _, s = scipy.signal.spectrogram(
        x,
        fs=SR,
        noverlap=hop_length,
        nperseg=win_length,
        nfft=n_fft,
        return_onesided=True,
        scaling="spectrum",
        mode="magnitude",
        axis=-1,
    )

    # Get amplitude values between 0 and 1 logarithmically spaced between a
    # maximum reference value and a minimum value (similarly to what is done when converting
    # to dB).
    return np.log10((s / ref + amin) / amin) / np.log10(ref / amin)


def make_melspectrogram(
    x, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_fft=N_FFT, **kwargs
):
    s = make_spectrogram(x, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
    return lbr.feature.melspectrogram(S=s, sr=SR, n_mels=N_MELS, fmax=FMAX, fmin=FMIN)


def make_mfcc(
    x, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_fft=N_FFT, mfcc=True, **kwargs
):
    s = make_melspectrogram(
        x, hop_length=hop_length, win_length=win_length, n_fft=n_fft
    )

    s = lbr.feature.mfcc(
        S=s,
        sr=SR,
        n_mfcc=N_MFCC,
    )
    d = lbr.feature.delta(s)
    d2 = lbr.feature.delta(s, order=2)
    if mfcc:
        return np.vstack([s, d, d2])
    return np.vstack([d, d2])


def make_old_mfcc(x, mfcc=True, **kwargs):
    s = lbr.feature.mfcc(
        y=x,
        sr=SR,
        n_mfcc=20,
        fmin=500,
        fmax=8000,
        hop_length=160,
        win_length=320,
        lifter=0,
    )
    d = lbr.feature.delta(s, mode="nearest")
    d2 = lbr.feature.delta(s, order=2, mode="nearest")
    if mfcc:
        return np.vstack([s, d, d2])
    return np.vstack([d, d2])


def preprocess(
    dataset,
    max_length=MAX_LENGTH,
    features="deltas",
    norm_audio=True,
    **transform_params,
):
    @delayed
    def transform(d):
        if isinstance(d, PathLike) or isinstance(d, str):
            y = lbr.load(d, sr=None)[0]
        else:
            y = d

        if norm_audio:
            y = lbr.util.normalize(y, axis=-1)

        if features == "deltas":
            return make_mfcc(y[:max_length], **transform_params)
        elif features == "spec":
            return make_spectrogram(y[:max_length], **transform_params)
        elif features == "mel":
            return make_melspectrogram(y[:max_length], **transform_params)
        elif features == "base":
            return make_old_mfcc(y[:max_length], **transform_params)
        else:
            raise NotImplementedError(features)

    results = Parallel(n_jobs=-1)(transform(d) for d in dataset)

    return results


def get_spec(df, feat="feat"):
    return np.dstack(df[feat].values)


def get_features(df, feat="feat"):
    return np.concatenate(get_spec(df, feat=feat)).T


def classify(estimator, name, rdf, gdf, feature="feat"):

    if feature.startswith("n_"):
        prefix = "n_"
    else:
        prefix = ""

    print(f"Training {name} on {feature}")

    X_train = get_features(rdf.query("split=='train'"), feat=feature)
    y_train = rdf.query("split=='train'")["label"]

    print(X_train.shape)

    X_test = get_features(rdf.query("split=='test'"), feat=feature)
    y_test = rdf.query("split=='test'")["label"]

    tic = time.time()

    y_pred = estimator.fit(X_train, y_train).predict(X_test)

    toc = time.time()
    print("Elapsed: ", toc - tic)

    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy: ", acc)

    cm = confusion_matrix(y_test, y_pred, labels=estimator.classes_)

    print("Prediction on whole corpus")
    tic = time.time()

    X_gen = get_features(gdf, feat=feature)
    print(X_gen.shape)

    gdf[f"{prefix}{name}_y"] = estimator.predict(X_gen)

    toc = time.time()
    print("Elapsed: ", toc - tic)

    return cm, acc


def distance_to_mean(rdf, gdf, labels, clfs):

    mean_feats = [
        np.mean(
            get_features(rdf.query(f"label==@l"), feat="feat"), axis=0, keepdims=True
        )
        for l in labels
    ]
    mean_feats = np.vstack(mean_feats)

    mean_specs = [
        np.mean(
            get_features(rdf.query("label==@l"), feat="spec"), axis=0, keepdims=True
        )
        for l in labels
    ]
    mean_specs = np.vstack(mean_specs)

    mean_nfeats = [
        np.mean(
            get_features(rdf.query(f"label==@l"), feat="n_feat"), axis=0, keepdims=True
        )
        for l in labels
    ]
    mean_nfeats = np.vstack(mean_nfeats)

    gen_feats = get_features(gdf, feat="feat")
    gen_specs = get_features(gdf, feat="spec")
    gen_nfeats = get_features(gdf, feat="n_feat")

    for name in clfs:
        d_f = np.full(len(gdf), np.nan)
        d_s = np.full(len(gdf), np.nan)
        d_nf = np.full(len(gdf), np.nan)

        c_f = np.full(len(gdf), np.nan)
        c_s = np.full(len(gdf), np.nan)
        c_nf = np.full(len(gdf), np.nan)

        for i, label in enumerate(labels):
            m_feat = mean_feats[i, None]
            m_spec = mean_specs[i, None]
            m_nfeat = mean_nfeats[i, None]

            idxs = gdf.query(f"{name}_y==@label").index.values
            nidxs = gdf.query(f"n_{name}_y==@label").index.values

            d_feat = cdist(gen_feats, m_feat)
            d_spec = cdist(gen_specs, m_spec)
            d_nfeat = cdist(gen_nfeats, m_nfeat)

            d_f[idxs] = d_feat[idxs].flatten()
            d_s[idxs] = d_spec[idxs].flatten()
            d_nf[nidxs] = d_nfeat[nidxs].flatten()

            d_feat = cdist(gen_feats, m_feat, metric="correlation")
            d_spec = cdist(gen_specs, m_spec, metric="correlation")
            d_nfeat = cdist(gen_nfeats, m_nfeat, metric="correlation")

            c_f[idxs] = d_feat[idxs].flatten()
            c_s[idxs] = d_spec[idxs].flatten()
            c_nf[nidxs] = d_nfeat[nidxs].flatten()

        gdf[f"d_{name}_feat"] = d_f
        gdf[f"d_{name}_spec"] = d_s
        gdf[f"d_{name}_nfeat"] = d_nf

        gdf[f"c_{name}_feat"] = c_f
        gdf[f"c_{name}_spec"] = c_s
        gdf[f"c_{name}_nfeat"] = c_nf

    # for i, label in enumerate(laels):
    #     m_feat = mean_feats[i, None]
    #     m_spec = mean_specs[i, None]
    #     m_nfeat = mean_nfeats[i, None]

    #     d_feat = cdist(gen_feats, m_feat)
    #     d_spec = cdist(gen_specs, m_spec)
    #     d_nfeat = cdist(gen_nfeats, m_nfeat)

    #     gdf[f"d_feat_{label}"] = d_feat.flatten()
    #     gdf[f"d_spec_{label}"] = d_spec.flatten()
    #     gdf[f"d_nfeat_{label}"] = d_nfeat.flatten()

    #     d_feat = cdist(gen_feats, m_feat, metric="correlation")
    #     d_spec = cdist(gen_specs, m_spec, metric="correlation")
    #     d_nfeat = cdist(
    #         gen_nfeats, m_nfeat, metric="correlation"
    #     )

    #     gdf[f"corr_feat_{label}"] = d_feat.flatten()
    #     gdf[f"corr_spec_{label}"] = d_spec.flatten()
    #     gdf[f"corr_nfeat_{label}"] = d_nfeat.flatten()

    return gdf


def mean_spectrogram(gdf, col, labels, save_to):

    n = len(labels)
    n_cols = 4
    n_rows = n // 4 + int(n % 4 > 0)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    cond = gdf["cond"].unique()[0]
    fig.suptitle(cond)

    for i, lbl in enumerate(labels):
        ax = axs[i]
        q = gdf.query(f"{col}==@lbl")
        if len(q) == 0:
            continue
        ax.imshow(
            np.mean(q["spec"].values, axis=0),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="magma",
            vmin=0.0,
        )
        ax.set_title(f"~{lbl} (n={len(q)})")

    save_file = save_to / f"{cond}-{col}.pdf"

    fig.savefig(save_file, bbox_inches="tight")


def make_name(args):
    params = json.dumps(vars(args), sort_keys=True, separators=(",", ":"))
    return uuid.uuid5(uuid.NAMESPACE_URL, params).hex


if __name__ == "__main__":

    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    save_dir_preds = save_dir / "predictions"
    save_dir_reports = save_dir / "classification_reports"
    save_dir_specs = save_dir / "mean_spectrograms"

    save_dir_preds.mkdir(parents=True, exist_ok=True)
    save_dir_reports.mkdir(parents=True, exist_ok=True)
    save_dir_specs.mkdir(parents=True, exist_ok=True)

    reports = vars(args)

    files = list(Path(args.data_dir).rglob("**/*.wav"))

    # --------------------------------------------
    # Real dataset

    data_table = {
        "file": files,
        "label": [f.parent.name for f in files],
        "kind": "real",
    }

    rdf = pd.DataFrame(data_table)
    rdf = rdf.sort_values(by=["label", "file"]).reset_index(drop=True)

    rdf["feat"] = preprocess(track(rdf.file), **vars(args))
    rdf["spec"] = preprocess(
        track(rdf.file),
        features="spec",
        ref=REF_POW,
        amin=MIN_POW,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
    )

    all_labels = sorted(rdf.label.unique().tolist())
    labels = [l for l in all_labels if "EARLY" not in l and "WN" not in l]

    split = np.array(["train"] * len(rdf))
    split[::10] = "test"

    rdf["split"] = split

    # --------------------------------------------
    # GAN dataset
    file = next(
        Path(args.gen_dir).rglob(
            f"version_{args.version}/*epoch_{str(args.epoch).zfill(3)}*.npz"
        )
    )
    archive = np.load(file)

    x = preprocess(track(archive["x"]), **vars(args))
    specs = preprocess(
        track(archive["x"]),
        features="spec",
        ref=REF_POW,
        amin=MIN_POW,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
    )

    data_table = {
        "file": str(file),
        "version": int(archive["version"]),
        "epoch": int(archive["epoch"]),
        "y": [a for a in archive["y_argmax"]],
        "z": [a for a in archive["z"]],
        "kind": "gen",
        **reports,
    }

    # gdf = pd.DataFrame(data_table).explode(["y", "z"])
    gdf = pd.DataFrame(data_table)

    gdf["feat"] = x
    gdf["spec"] = specs
    gdf["cond"] = (
        gdf["features"]
        + gdf["norm_audio"].apply(lambda x: "-norm_audio" if x else "")
        + gdf["mfcc"].apply(lambda x: "+mfcc" if x else "")
    )

    gdf = gdf.sort_values(by=["version", "epoch"]).reset_index(drop=True)

    # Stats per channel, for normalization
    mean_hz = np.mean(get_spec(rdf, feat="feat"), axis=(1, 2))
    std_hz = np.std(get_spec(rdf, feat="feat"), axis=(1, 2))

    rdf["n_feat"] = rdf["feat"].apply(
        lambda x: (x - mean_hz[:, np.newaxis]) / std_hz[:, np.newaxis]
    )
    gdf["n_feat"] = gdf["feat"].apply(
        lambda x: (x - mean_hz[:, np.newaxis] / std_hz[:, np.newaxis])
    )

    filename = Path(gdf["cond"].unique()[0])

    # --------------------------------------------
    # Classification

    clfs = {"knn": KNeighborsClassifier(n_neighbors=100), "svm": SVC(C=0.1)}

    # Unormalized features
    for name, clf in clfs.items():
        cm, acc = classify(clf, name, rdf, gdf, feature="feat")
        reports[name] = {"cm": cm.tolist(), "acc": float(acc)}

        mean_spectrogram(gdf, col=f"{name}_y", labels=labels, save_to=save_dir_specs)

    # Normalized features
    for name, clf in clfs.items():
        cm, acc = classify(clf, name, rdf, gdf, feature="n_feat")
        reports[name] = {"n_cm": cm.tolist(), "n_acc": float(acc)}

        mean_spectrogram(gdf, col=f"n_{name}_y", labels=labels, save_to=save_dir_specs)

    with (save_dir_reports / filename.with_suffix(".yml")).open("w+") as fp:
        yaml.dump(reports, fp)

    gdf = distance_to_mean(rdf, gdf, labels=labels, clfs=list(clfs.keys()))

    gdf.drop(["feat", "spec", "n_feat"], axis=1).to_csv(
        save_dir_preds / filename.with_suffix(".csv"), index=False
    )
