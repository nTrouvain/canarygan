from typing import Any, List, Dict
from abc import ABCMeta, abstractmethod
import dill as pickle

import numpy as np

from reservoirpy.nodes import ESN, Reservoir, Ridge
from .decoding import reduce_time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def build_reservoir(
    units=1000,
    dim=13,
    sr=0.5,
    lr=0.05,
    input_bias=True,
    mfcc_scaling=0.0,
    delta_scaling=1.0,
    delta2_scaling=0.7,
    bias_scaling=1.0,
    input_connectivity=0.1,
    rc_connectivity=0.1,
    use_mfcc=False,
    use_delta=True,
    use_delta2=True,
    seed=None,
    name="",
):
    scalings = []
    if use_mfcc:
        scalings.append(mfcc_scaling)
    if use_delta:
        scalings.append(delta_scaling)
    if use_delta2:
        scalings.append(delta2_scaling)

    if len(scalings) == 0:
        raise ValueError(
            "At least one of use_mfcc, use_delta or use_delta2 must be True."
        )

    iss = np.concatenate([[sc] * dim for sc in scalings])

    reservoir = Reservoir(
        units,
        input_dim=dim * len(scalings),
        lr=lr,
        sr=sr,
        input_bias=input_bias,
        input_scaling=iss,
        bias_scaling=bias_scaling,
        input_connectivity=input_connectivity,
        rc_connectivity=rc_connectivity,
        name=f"{'-'.join(['reservoir', name])}",
        seed=seed,
    )

    reservoir.initialize()

    return reservoir


class Decoder(metaclass=ABCMeta):

    model: Any
    class_labels: List[str]
    class_to_idx: Dict[str, int]

    @abstractmethod
    def fit(self, X, y):
        return self

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    def save(self, path):
        with open(path, "wb+") as fp:
            pickle.dump(self, fp)
        return self


class Unsupervised(Decoder, metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y=None):
        return self


class Supervised(Decoder, metaclass=ABCMeta):
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / y.shape[0]


class ESNDecoder(Supervised):

    @classmethod
    def from_reservoir(
        cls, reservoir, ridge=1e-8, reduce_time_policy="argmax", name=""
    ):
        decoder = cls()

        readout = Ridge(ridge=ridge, name=f"{'-'.join(['readout', name])}")

        esn = ESN(
            reservoir=reservoir,
            readout=readout,
            workers=-1,
            name=f"{'-'.join(['esn', name])}",
        )

        decoder.model = esn
        decoder.name = name
        decoder.reduce_time_policy = reduce_time_policy

        return decoder

    def __init__(
        self,
        units=1000,
        dim=13,
        sr=0.5,
        lr=0.05,
        input_bias=True,
        mfcc_scaling=0.0,
        delta_scaling=1.0,
        delta2_scaling=0.7,
        bias_scaling=1.0,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        ridge=1e-8,
        use_mfcc=False,
        use_delta=True,
        use_delta2=True,
        reduce_time_policy="argmax",
        seed=None,
        name="",
    ):
        super().__init__()

        reservoir = build_reservoir(
            units=units,
            dim=dim,
            sr=sr,
            lr=lr,
            input_bias=input_bias,
            mfcc_scaling=mfcc_scaling,
            delta_scaling=delta_scaling,
            delta2_scaling=delta2_scaling,
            bias_scaling=bias_scaling,
            input_connectivity=input_connectivity,
            rc_connectivity=rc_connectivity,
            use_mfcc=use_mfcc,
            use_delta=use_delta,
            use_delta2=use_delta2,
            seed=seed,
            name=name,
        )

        readout = Ridge(ridge=ridge, name=f"{'-'.join(['readout', name])}")

        esn = ESN(
            reservoir=reservoir,
            readout=readout,
            workers=-1,
            name=f"{'-'.join(['esn', name])}",
        )

        self.model = esn
        self.name = name
        self.reduce_time_policy = reduce_time_policy

    def fit(self, X, y):

        if X.ndim == 2:
            X = X[np.newaxis, ...]

        y = y.flatten()

        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X.shape[0] != y.shape[0] ({X.shape[0]} != {y.shape[0]})")

        n, t, _ = X.shape

        labels = sorted(np.unique(y).tolist())

        # One-hot encode
        idx_map = {lbl: i for i, lbl in enumerate(labels)}
        y_idx = np.array([idx_map[i] for i in y])
        y_encoded = np.zeros((n, t, len(labels)))
        y_encoded[np.arange(n), :, y_idx] = 1

        self.model.fit(X, y_encoded)

        self.class_labels = labels
        self.class_to_idx = idx_map

        return self

    def predict_proba(self, X):

        if X.ndim == 2:
            X = X[np.newaxis, ...]

        y = self.model.run(X)

        if isinstance(y, list):
            y = np.r_[y]

        if y.ndim == 2:
            y = y[np.newaxis, ...]

        y = reduce_time(y, axis=1, policy=self.reduce_time_policy)

        return y

    def predict(self, X):
        y_pred = self.predict_proba(X)

        y_pred = np.argmax(y_pred, axis=1)

        idx_map = {i: lbl for lbl, i in self.class_to_idx.items()}
        y_pred = np.array([idx_map[y] for y in y_pred])

        return y_pred


class SklearnDecoder(Decoder):
    def fit(self, X, y):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self.model.fit(X, y)

        self.class_labels = self.model.classes_
        self.class_to_idx = {k: i for i, k in enumerate(self.class_labels)}

        return self

    def predict_proba(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)

    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)


class KNNDecoder(SklearnDecoder):
    def __init__(self, n_neighbors=100):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)


class SVMDecoder(SklearnDecoder):
    def __init__(self, C=0.1):
        super().__init__()
        self.model = SVC(C=C)
