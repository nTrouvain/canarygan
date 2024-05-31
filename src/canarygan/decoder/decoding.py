from multiprocessing import Value
import numpy as np

from joblib import delayed, Parallel

from .preprocessing import make_mfcc
from .dataset import DecoderDataset


@delayed
def preprocess_sound(y, sampling_rate, hop_length, win_length, lifter, n_mfcc, padding):
    features = make_mfcc(
        y, sampling_rate, hop_length, win_length, lifter, n_mfcc, padding
    )
    return np.concatenate(features[1:], axis=1)


def decode(
    decoder,
    x,
):
    # Used to get preprocessing parameters
    d = DecoderDataset()

    with Parallel(n_jobs=-1) as parallel:
        x_mfcc = parallel(
            [
                preprocess_sound(
                    xi,
                    d.sampling_rate,
                    d.hop_length,
                    d.win_length,
                    d.lifter,
                    d.n_mfcc,
                    d.padding,
                )
                for xi in x
            ]
        )

    y = decoder.run(x_mfcc)

    if isinstance(y, list):
        y = np.r_[y]

    return y


def select_max(y):
    """From Silvia's code.
    'Select the peak of the most active output' actually means
    keeping activity only at the timestep where peak activity
    (for any classes) happened.
    """

    def select(y_i):
        t_max, _ = np.unravel_index(np.argmax(y_i), y_i.shape)
        return y_i[t_max]

    return np.vstack([select(y_i) for y_i in y])


def reduce_time(y, axis=1, policy="max"):
    if policy == "max":
        return np.max(y, axis=axis)
    elif policy == "sum":
        return np.sum(y, axis=axis)
    elif policy == "mean":
        return np.mean(y, axis=axis)
    elif policy == "argmax":
        if axis != 1:
            raise ValueError("Can't apply argmax policy with axis != 1")
        return select_max(y)
    else:
        raise NotImplementedError(f"policy: {policy}")


def maximum_a_posteriori(logits, axis=1):
    # Reduce time dimension
    y_pred = np.argmax(logits, axis=axis)
    return y_pred
