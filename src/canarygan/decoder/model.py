# Author: Nathan Trouvain at 10/20/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np

from reservoirpy.nodes import ESN, Reservoir, Ridge


def esn(
    units=1000,
    dim=20,
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
    )

    readout = Ridge(ridge=ridge, name=f"{'-'.join(['readout', name])}")

    esn = ESN(
        reservoir=reservoir,
        readout=readout,
        workers=-1,
        name=f"{'-'.join(['esn', name])}",
    )

    return esn
