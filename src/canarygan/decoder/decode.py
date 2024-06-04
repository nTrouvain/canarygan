# Author: Nathan Trouvain <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Decoding functions, from audio and decoder instance.
"""
import numpy as np

from .preprocess import preprocess
from .dataset import DecoderDataset


def select_argmax(y):
    """
    Select timestep where activity reached a peak
    in series of signal y.

    From Pagliarini et al. (2018).

    'Select the peak of the most active output' actually means
    keeping activity only at the timestep where peak activity
    (for any classes) happened.
    """

    def select(y_i):
        t_max, _ = np.unravel_index(np.argmax(y_i), y_i.shape)
        return y_i[t_max]

    return np.vstack([select(y_i) for y_i in y])


def reduce_time(y, axis=1, policy="max"):
    """
    Aggregate decoder response over time to obtain a
    single prediction vector per audio sample. Useful for
    RNN-based decoders, like ESN.

    Parameters
    ----------
    y : np.ndarray
        Array of shape (sample, timesteps, classes) containing
        one prediction per timestep for each sample of length timesteps.
    axis : int, default to 1
        Time axis in y.
    policy : ["max", "sum", "mean", "argmax"], default to "max"
        Method used to perform aggregation:
            - max: take maximum values over time
            - sum: sum response over time
            - mean: take mean values over time
            - argmax: select timestep where response reached a peak (from Pagliarini et al (2018)).

    Returns:
    --------
    np.ndarray
        Array of shape (sample, classes).
    """
    if policy == "max":
        return np.max(y, axis=axis)
    elif policy == "sum":
        return np.sum(y, axis=axis)
    elif policy == "mean":
        return np.mean(y, axis=axis)
    elif policy == "argmax":
        if axis != 1:
            raise ValueError("Can't apply argmax policy with axis != 1")
        return select_argmax(y)
    else:
        raise NotImplementedError(f"policy: {policy}")


def maximum_a_posteriori(logits, axis=1):
    """
    Select index of maximum probability in a vector of
    predictions.

    Parameters
    ----------
    logits : np.ndarray
        Array of predictions of shape (samples, classes).
    axis : np.ndarray
        Axis from which to select index.

    Returns
    -------
    np.ndarray
        Array of shape (samples,)
    """
    y_pred = np.argmax(logits, axis=axis)
    return y_pred


def decode(
    decoder,
    x,
    transform_params=None,
):
    """
    Preprocess and decode canary syllable samples.

    Parameters
    ----------
    decoder : canarygan.decoder.Decoder
        A Decoder instance.
    x : np.ndarray or list of str
        Audio data to decode. If list of str, will consider each
        string as a path and load audio from path.
    transform_params : dict or str, optional
        Preprocessing parameters. If str, will consider it as a path
        and load the file as a canarygan.decoder.dataset.Dataset parameter
        checkpoint file in YAML format. If None, will use
        canarygan.decoder.dataset.Dataset default values.

    Returns
    -------
    np.ndarray
        Array of shape (samples, classes) containing decoded syllable class
        for each audio sample.
    """
    # Used to get preprocessing parameters
    d = DecoderDataset()

    if transform_params is not None and not isinstance(transform_params, dict):
        d.load_params(transform_params)
        transform_params = d.transform_kwargs
    elif transform_params is None:
        transform_params = d.transform_kwargs

    x = preprocess(x, **transform_params)

    if isinstance(x, list):
        x = np.r_[x]

    if hasattr(decoder, "predict"):
        y = decoder.predict(x)
    elif hasattr(decoder, "run"):
        y = decoder.run(x)  # kept for compatibility with early experiments.
    else:
        raise ValueError(
            "Decoder is not a Decoder instance (no 'predict' or 'run' method)."
        )

    if isinstance(y, list):
        y = np.r_[y]

    return y
