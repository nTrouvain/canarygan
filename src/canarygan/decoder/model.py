# Author: Nathan Trouvain <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Models of decoder.
"""
from pathlib import Path
from typing import Any, List, Dict
from abc import ABCMeta, abstractmethod
import dill as pickle

import numpy as np

from reservoirpy.nodes import ESN, Reservoir, Ridge
from .decode import reduce_time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_decoder(decoder_ckpt):
    """Load a decoder object from a saved checkpoint."""
    decoder_ckpt = Path(decoder_ckpt)
    if not decoder_ckpt.is_file():
        raise FileNotFoundError(f"Decoder: {decoder_ckpt}")

    with open(decoder_ckpt, "rb") as fp:
        decoder = pickle.load(fp)

    return decoder


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
    """
    Create a reservoir of neurons, without readout.

    Default values come from Pagliarini et al. (2018).

    Parameters
    ----------
    units : int, default to 1000
        Number of neurons
    dim : int, default to 13
        Dimension of input features (MFCC dimension in
        Pagliarini et al. (2018)).
    sr : float, default to 0.5
        Spectral radius of connectivity matrix.
    lr : float, default to 0.05
        Leak rate of neurons.
    input_bias : bool, default to True
        If True, add a constant term to inputs.
    mfcc_scaling : float, default to 0.0
        Input scaling applied to MFCC (first 13 input features).
    delta_scaling : float, default to 1.0
        Input scaling applied to MFCC derivatives (next 13 coefficients).
    delta2_scaling : float, default to 0.7
        Input scaling applied to MFCC second order derivatives (last 13 coefficients).
    bias_scaling : float, default to 1.0,
        Input scaling applied to constant term (if input_bias is True).
    input_connectivity : float, default to 0.1
        Density of connection between input neurons and reservoir.
    rc_connectivity : float, default ot 0.1
        Density of connection between reservoir's neurons
    use_mfcc : bool, default to False
        If True, will connect MFCC inputs to reservoir.
    use_delta : bool, default to True
        If True, will connect MFCC derivatives to reservoir.
    use_delta2 : bool, default to True
        If True, will connect MFCC second order derivatives to reservoir.
    seed : int, optional
        Random seed for reservoir connections.
    name : str, default to ""
        Model's name.

    Returns
    -------
    reservoirpy.nodes.Reservoir
        An initialized Reservoir instance.
    """
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
    """
    Base class for decoder objects.
    """

    model: Any
    class_labels: List[str]
    class_to_idx: Dict[str, int]

    @abstractmethod
    def fit(self, X, y):
        """
        Fit decoder to data.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (samples, timesteps, features) containing audio
            signals to decode.
        y : np.ndarray
            Array of shape (samples, ) containing syllable labels to predict.

        Returns
        -------
        Decoder
            Self.
        """
        return self

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    """
    Decode audio data.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (samples, timesteps, features) containing audio
        signals to decode.

    Returns
    -------
    np.ndarray
        Array of shape (samples, ) containing syllable labels.
    """

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    """
    Same as predict, but returns a per-class probability score.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (samples, timesteps, features) containing audio
        signals to decode.

    Returns
    -------
    np.ndarray
        Array of shape (samples, classes) containing scores per syllable class.
    """

    def save(self, path):
        """
        Save this decoder instance.

        Will use pickle or dill to do so.

        Parameters
        ----------
        path : str
            File name. If the file already exists, it will be
            overwitten.

        Returns
        -------
        Decoder
            Self.
        """
        with open(path, "wb+") as fp:
            pickle.dump(self, fp)
        return self


class Unsupervised(Decoder, metaclass=ABCMeta):
    """
    Base class for unsupervised decoders.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit decoder to data.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (samples, timesteps, features) containing audio
            signals to decode.
        y : np.ndarray, optional
            Will be ignored, kept for consistency.

        Returns
        -------
        Decoder
            Self.
        """
        return self


class Supervised(Decoder, metaclass=ABCMeta):
    """
    Base class for supervised decoders.
    """

    def score(self, X, y):
        """
        Compute accuracy score of decoder.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (samples, timesteps, features) containing audio
            signals to decode.
        y : np.ndarray
            Array of shape (samples, ) containing ground truth labels.

        Returns
        -------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / y.shape[0]


class ESNDecoder(Supervised):
    """
    Echo State Network decoder.
    """

    @classmethod
    def from_reservoir(
        cls, reservoir, ridge=1e-8, reduce_time_policy="argmax", name=""
    ):
        """
        Create an ESN from a previously built reservoir.

        Parameters
        ----------
        reservoir : reservoirpy.nodes.Reservoir
            A Reservoir instance.
        ridge : float, default to 1e-8
           Readout regularization parameter.
        reduce_time_policy : ["max", "sum", "mean", "argmax"], default to "argmax"
            How readout activity is aggregated along time axis before giving
            the final prediction. See reduce_time documentation.
        name : str, default to ""
            Model's name.

        Returns
        -------
        ESNDecoder
            A new ESNDecoder object.
        """
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
        """
        Create an ESNDecoder object.

        Default values come from Pagliarini et al. (2018).

        Parameters
        ----------
        units : int, default to 1000
            Number of neurons
        dim : int, default to 13
            Dimension of input features (MFCC dimension in
            Pagliarini et al. (2018)).
        sr : float, default to 0.5
            Spectral radius of connectivity matrix.
        lr : float, default to 0.05
            Leak rate of neurons.
        input_bias : bool, default to True
            If True, add a constant term to inputs.
        mfcc_scaling : float, default to 0.0
            Input scaling applied to MFCC (first 13 input features).
        delta_scaling : float, default to 1.0
            Input scaling applied to MFCC derivatives (next 13 coefficients).
        delta2_scaling : float, default to 0.7
            Input scaling applied to MFCC second order derivatives (last 13 coefficients).
        bias_scaling : float, default to 1.0,
            Input scaling applied to constant term (if input_bias is True).
        input_connectivity : float, default to 0.1
            Density of connection between input neurons and reservoir.
        rc_connectivity : float, default ot 0.1
            Density of connection between reservoir's neurons
        use_mfcc : bool, default to False
            If True, will connect MFCC inputs to reservoir.
        use_delta : bool, default to True
            If True, will connect MFCC derivatives to reservoir.
        use_delta2 : bool, default to True
            If True, will connect MFCC second order derivatives to reservoir.
        ridge : float, default to 1e-8
           Readout regularization parameter.
        reduce_time_policy : ["max", "sum", "mean", "argmax"], default to "argmax"
            How readout activity is aggregated along time axis before giving
            the final prediction. See reduce_time documentation.
        seed : int, optional
            Random seed for reservoir connections.
        name : str, default to ""
            Model's name.
        """

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


class SklearnDecoder(Decoder, metaclass=ABCMeta):
    """
    Base class for scikit-learn tool based decoders.
    """

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
    """
    k-nearest neighbors decoder.
    """

    def __init__(self, n_neighbors=100):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)


class SVMDecoder(SklearnDecoder):
    """
    SVM decoder.
    """

    def __init__(self, C=0.1):
        super().__init__()
        self.model = SVC(C=C)
