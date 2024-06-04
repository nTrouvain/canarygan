import logging
import sys
import argparse
import uuid
import dill as pickle

from pathlib import Path

import numpy as np
import optuna
import reservoirpy as rpy
from sklearn.model_selection import StratifiedKFold

from pmseq.decoder.base import ESNDecoder, build_reservoir
from pmseq.decoder.dataset import DecoderDataset


rpy.verbosity(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "program",
    type=str,
    choices=["optim", "optim_ridge", "create_reservoirs"],
    default="optim",
)
parser.add_argument("--preprocessed_dir", type=str)
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--n_instances", type=int, default=5)
parser.add_argument("--instance_dir", type=str, default=None)
parser.add_argument("--storage", type=str, default="optuna.log")

args = parser.parse_args()

SAMPLING_RATE = 16000
MAX_TIME = round(0.3 * SAMPLING_RATE)


def objective(trial: optuna.Trial):

    n_splits = 5
    n_instances = args.n_instances

    sr = trial.suggest_float("sr", low=1e-3, high=10.0, log=True)
    lr = trial.suggest_float("lr", low=1e-4, high=1.0, log=True)
    delta_scaling = trial.suggest_float("d_is", low=1e-2, high=1e3, log=True)
    delta2_scaling = trial.suggest_float("d2_is", low=1e-2, high=1e3, log=True)
    ridge = trial.suggest_float("ridge", low=1e-8, high=1e-3, log=True)

    dataset = DecoderDataset.from_checkpoint(Path(args.preprocessed_dir))

    models = []
    for i in range(n_instances):
        model = ESNDecoder(
            name=uuid.uuid4().hex + f"_{i}",
            units=1000,
            sr=sr,
            lr=lr,
            delta_scaling=delta_scaling,
            delta2_scaling=delta2_scaling,
            ridge=ridge,
        )
        models.append(model)

    x_train, y_train = dataset.train_data

    print(x_train.shape)

    # Decimate dataset 10x
    n = x_train.shape[0]
    idxs = np.arange(0, n, 10)
    x_train = x_train[idxs]
    y_train = y_train[idxs]

    n_splits = 5
    split = StratifiedKFold(n_splits=n_splits).split(
        np.arange(len(x_train)).reshape(-1, 1), y_train
    )

    accuracy = 0
    for train_idx, test_idx in split:
        x_fold, y_fold = x_train[train_idx], y_train[train_idx]
        x_val, y_val = x_train[test_idx], y_train[test_idx]

        for model in models:
            acc = model.fit(x_fold, y_fold).score(x_val, y_val)
            accuracy += acc

    accuracy /= n_splits * n_instances

    print(accuracy)

    return accuracy


def ridge_seed_objective(trial: optuna.Trial):

    n_splits = 5

    ridge = trial.suggest_float("ridge", low=1e-8, high=10.0, log=True)
    seed = trial.suggest_categorical(
        "seed",
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        ],
    )

    dataset = DecoderDataset.from_checkpoint(Path(args.preprocessed_dir))

    instance = list(Path(args.instance_dir).glob(f"*seed_{seed}.reservoir.pkl"))[0]

    with open(instance, "rb") as fp:
        reservoir = pickle.load(fp)

    model = ESNDecoder.from_reservoir(
        reservoir,
        name=uuid.uuid4().hex + f"_{seed}",
        ridge=ridge,
    )

    x_train, y_train = dataset.train_data

    print(x_train.shape)

    # Decimate dataset 10x
    n = x_train.shape[0]
    idxs = np.arange(0, n, 10)
    x_train = x_train[idxs]
    y_train = y_train[idxs]

    n_splits = 5
    split = StratifiedKFold(n_splits=n_splits).split(
        np.arange(len(x_train)).reshape(-1, 1), y_train
    )

    accuracy = 0
    for train_idx, test_idx in split:
        x_fold, y_fold = x_train[train_idx], y_train[train_idx]
        x_val, y_val = x_train[test_idx], y_train[test_idx]

        acc = model.fit(x_fold, y_fold).score(x_val, y_val)
        accuracy += acc

    accuracy /= n_splits

    print(accuracy)

    return accuracy


def create_reservoirs():
    sr = 0.02
    lr = 0.05
    delta_scaling = 30.0
    delta2_scaling = 1.0

    n_instances = args.n_instances

    seeds = np.arange(n_instances)

    instance_dir = Path(args.instance_dir)
    instance_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_instances):
        name = uuid.uuid4().hex + f"-seed_{seeds[i]}"

        reservoir = build_reservoir(
            units=1000,
            sr=sr,
            lr=lr,
            delta_scaling=delta_scaling,
            delta2_scaling=delta2_scaling,
            seed=seeds[i],
            name=name,
        )

        with (instance_dir / f"{name}.reservoir.pkl").open("wb+") as fp:
            pickle.dump(reservoir, fp)


if args.program == "optim":

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            args.storage
        ),  # NFS path for distributed optimization
    )

    study = optuna.create_study(
        study_name="decoder-6",
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
        load_if_exists=True,
        storage=storage,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))

elif args.program == "optim_ridge":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            args.storage
        ),  # NFS path for distributed optimization
    )

    study = optuna.create_study(
        study_name="ridge-1",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
        storage=storage,
    )
    study.optimize(ridge_seed_objective, n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))

elif args.program == "create_reservoirs":
    create_reservoirs()
