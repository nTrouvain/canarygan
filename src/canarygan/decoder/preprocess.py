# Author: Nathan Trouvain <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Audio data preprocessing functions for decoders.
"""
from os import PathLike

import librosa as lbr
import numpy as np
import scipy.signal

from joblib import delayed, Parallel


# Reference power: power is scaled wrt to this value
REF_POW = 1.0
# Min power: power is shifted by this amount to fit in the logarithmic scale
MIN_POW = 1e-4
# Default sampling rate
SR = 16000


def load(audio_file, sampling_rate):
    """
    Load audio from file and resample if necessary.
    """
    y, _ = lbr.load(audio_file, sr=sampling_rate)
    return y


def pad_or_cut(y, max_steps):
    """
    Cut audio sample to max_steps length or zero pad it.
    """
    if max_steps != -1:
        if y.shape[0] < max_steps:
            filler = np.zeros((max_steps,))
            filler[: y.shape[0]] = y.flatten()
            y = filler
        elif y.shape[0] > max_steps:
            y = y[:max_steps]
    return y


def make_spectrogram(x, sampling_rate, hop_length, win_length, n_fft=1024, **kwargs):
    """
    Transform audio signal into its normalized spectrogram.

    Parameters
    ----------
    x : np.ndarray
        Audio signal of shape (timesteps,).
    sampling_rate : int
        Audio sampling rate in Hz.
    hop_length : int
        Spectrogram window strides, in timesteps.
    win_length : int
        Spectrogram window length, in timesteps.
    n_fft : int, default to 1024
        Number of FFT bins computed.

    Returns
    -------
    np.ndarray
        Array of shape (timebins, frequency bins=n_fft // 2).
    """
    _, _, s = scipy.signal.spectrogram(
        x,
        fs=sampling_rate,
        window="hann",
        nperseg=win_length,
        noverlap=hop_length,
        nfft=n_fft,
        return_onesided=True,
        scaling="spectrum",
        axis=-1,
        mode="magnitude",
    )
    # Keep everything bounded between 0 and 1 to remain machine learning friendly
    return np.log10(s / (REF_POW * MIN_POW) + 1) / np.log10(1 / (REF_POW * MIN_POW) + 1)


def make_melspectrogram(x, n_mels=128, **kwargs):
    """
    Transform spectrogram scale from Hertz to Mels.

    Parameters
    ----------
    x : np.ndarray
        Audio signal of shape (timesteps,).
    n_mels : int, default to 128
        Number of Mel bins.
    **kwargs : optional
        Parameters for make_spectrogram.

    Returns
    -------
    np.ndarray
        Array of shape (timebins, n_mels).
    """
    s = make_spectrogram(x, **kwargs)
    return lbr.feature.melspectrogram(
        S=s, sr=kwargs.get("sampling_rate", SR), n_mels=n_mels
    )


def make_mfcc(x, mfcc, lifter, n_mfcc, **kwargs):
    """
    Extract MFCC from audio signal.

    Parameters
    ----------
    x : np.ndarray
        Audio signal of shape (timesteps,).
    mfcc : bool
        If True, will return MFCC. If False, will only return MFCC
        derivatives and second order derivatives.
    lifter : int
        Liftering parameters, to rescale MFCC.
    n_mfcc : int
        Number of coefficients to compute.
    **kwargs : optional
        Parameters from make_melspectrogram

    Returns
    -------
    np.ndarray
        Array of shape (timebins, n_mfcc * 2) if mfcc is False
        or (timebins, n_mfcc * 3) if mfcc is True. Features are
        inserted in this order, column-wise: MFCC (optional),
        first derivatives, second derivatives.
    """
    s = make_melspectrogram(x, **kwargs)
    s = lbr.feature.mfcc(
        S=s, sr=kwargs.get("sampling_rate", SR), lifter=lifter, n_mfcc=n_mfcc
    )
    d = lbr.feature.delta(s)
    d2 = lbr.feature.delta(s, order=2)
    if mfcc:
        return np.vstack([s, d, d2]).T
    return np.vstack([d, d2]).T


def make_mfcc_classic(
    x,
    mfcc,
    lifter,
    n_mfcc,
    padding,
    hop_length,
    win_length,
    fmin,
    fmax,
    n_fft=1024,
    **kwargs,
):
    """
    Extract MFCC from audio signal.

    Original code from Pagliarini et al. (2018).
    Spectrogram computations may be slightly inexact
    compared to the one performed by scipy.

    Parameters
    ----------
    x : np.ndarray
        Audio signal of shape (timesteps,).
    mfcc : bool
        If True, will return MFCC. If False, will only return MFCC
        derivatives and second order derivatives.
    lifter : int
        Liftering parameters, to rescale MFCC.
    n_mfcc : int
        Number of coefficients to compute.
    padding : str
        Padding mode for derivative computation.
    hop_length : int
        Spectrogram window strides, in timesteps.
    win_length : int
        Spectrogram window length, in timesteps.
    fmin : float
        Lower frequency boundary.
    fmax : float
        Upper frequency boundary.
    n_fft : int, default to 1024
        Number of FFT bins computed.

    Returns
    -------
    np.ndarray
        Array of shape (timebins, n_mfcc * 2) if mfcc is False
        or (timebins, n_mfcc * 3) if mfcc is True. Features are
        inserted in this order, column-wise: MFCC (optional),
        first derivatives, second derivatives.
    """
    s = lbr.feature.mfcc(
        y=x,
        sr=kwargs.get("sampling_rate", SR),
        lifter=lifter,
        n_fft=n_fft,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )
    d = lbr.feature.delta(s, mode=padding)
    d2 = lbr.feature.delta(s, order=2, mode=padding)
    if mfcc:
        return np.vstack([s, d, d2]).T
    return np.vstack([d, d2]).T


def preprocess(
    dataset,
    features="deltas",
    norm_audio=True,
    max_length=1.0,
    **transform_params,
):
    """
    Preprocess audio data for decoding.

    Parameters
    ----------
    dataset : list of str or np.ndarray or list of np.ndarray
        Data to transform. If list of str, will consider each
        string as a path and load audio from path.
    features : ["deltas", "spec", "mel", "mfcc-classic"], default to deltas
        Type of features to extract:
            - deltas: MFCC and derivatives
            - spec: spectrogam
            - mel: mel-spectrogam
            - mfcc-classic: MFCC and derivatives as done in Pagliarini et al. (2018).
    norm_audio : bool, default to True
        If True, will normalize audio values prior to preprocessing.
    max_length : float, default to 1.0
        Max length of audio samples, in second. Will pad or cut samples
        to reach that value.
    **transform_params
        Parameters for tranform functions.

    Returns
    -------
    list of np.ndarray
        Preprocessed samples.
    """

    sampling_rate = transform_params.get("sampling_rate", SR)
    max_steps = round(sampling_rate * max_length)

    @delayed
    def transform(d):
        if isinstance(d, PathLike) or isinstance(d, str):
            y = load(
                d,
                sampling_rate=sampling_rate,
            )
        else:
            y = d

        y = pad_or_cut(y, max_steps)

        if norm_audio:
            y = lbr.util.normalize(y, axis=-1)

        if features == "deltas":
            return make_mfcc(y, **transform_params)
        elif features == "spec":
            return make_spectrogram(y, **transform_params)
        elif features == "mel":
            return make_melspectrogram(y, **transform_params)
        elif features == "mfcc-classic":
            return make_mfcc_classic(y, **transform_params)
        else:
            raise NotImplementedError(features)

    results = Parallel(n_jobs=-1)(transform(d) for d in dataset)

    return results
