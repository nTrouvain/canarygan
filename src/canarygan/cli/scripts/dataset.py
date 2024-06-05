import pathlib
import json

import click
import numpy as np


from ...decoder import DecoderDataset


@click.command(
    name="sample", help="Randomly sample GAN latent space and save vectors to disk."
)
@click.option(
    "-s",
    "--save-dir",
    required=True,
    type=click.Path(),
    help="Directory where vectors will be stored.",
)
@click.option(
    "-n", "--n-samples", type=int, default=16000, help="Number of samples to draw."
)
@click.option("-d", "--dim", type=int, default=3, help="Latent space dimension.")
@click.option(
    "--dist",
    type=str,
    default="uniform",
    help="Distribution to sample values from. Must be a scipy.stats distribution.",
)
@click.option(
    "--dist-params",
    type=str,
    default='{"low": -1, "high": 1}',
    help="Distribution parameters, in JSON format.",
)
@click.option("--seed", type=int, default=0, help="Random state seed.")
def sample_z(
    save_dir,
    n_samples,
    dim=3,
    dist="uniform",
    dist_params={"low": -1, "high": 1},
    seed=0,
):
    rng = np.random.default_rng(seed)
    size = (n_samples, dim)
    rvs = getattr(rng, dist)

    if isinstance(dist_params, str):
        dist_params = json.loads(dist_params)

    print(f"Generating array of size {size}.")
    z = rvs(**dist_params, size=size)

    save_dir = pathlib.Path(save_dir)

    (save_dir / f"latent_z-dim_{dim}.npy").unlink(missing_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"latent_z-dim_{dim}.npy", z)

    return


_epilog = """
Availble features:
\n\t- deltas: MFCC and derivatives
\n\t- spec: spectrogam
\n\t- mel: mel-spectrogam
\n\t- mfcc-classic: MFCC and derivatives as done in Pagliarini et al. (2018).
"""


@click.command(
    name="build-decoder-dataset",
    help="Preprocess dataset for decoder training.",
    epilog=_epilog,
)
@click.option(
    "-d",
    "--data-dir",
    required=True,
    type=str,
    help="Dataset directory. Must contain WAV files.",
)
@click.option(
    "-s",
    "--save-dir",
    required=True,
    type=str,
    help="Destination directory, where preprocessed data will be stored.",
)
@click.option(
    "--features",
    type=click.Choice(["deltas", "spec", "mel", "mfcc-classic"]),
    default="deltas",
    help="Type of preprocessing.",
)
@click.option(
    "--n-mfcc",
    type=int,
    default=13,
    help="Number of MFCC to compute, when using 'deltas' or 'mfcc-classic'.",
)
@click.option(
    "--split",
    type=float,
    default=0.1,
    help="Proportion of data used as testing material.",
)
@click.option(
    "--sampling_rate", type=int, default=16000, help="Audio sampling rate, in Hz."
)
@click.option(
    "--norm-audio",
    is_flag=True,
    help="If used, will normalize audio values prior to preprocessing.",
)
@click.option(
    "--max-length",
    type=float,
    default=1.0,
    help="Max length of audio samples, in second. Will pad or cut samples to reach that value.",
)
@click.option(
    "--mfcc/--no-mfcc",
    default=False,
    help="If used, will return MFCC. Else, will only return MFCC derivatives and second order derivatives.",
)
@click.option(
    "--lifter", type=int, default=0, help="Liftering parameter, to rescale MFCC."
)
@click.option(
    "--padding",
    type=str,
    default="interp",
    help="Padding mode for derivative computation.",
)
@click.option(
    "--hop-length",
    type=int,
    default=128,
    help="Spectrogram window strides, in timesteps.",
)
@click.option(
    "--win-length",
    type=int,
    default=256,
    help="Spectrogram, window length, in timesteps.",
)
@click.option(
    "--fmin", type=float, default=500, help="Lower frequency boundary for MFCC, in Hz."
)
@click.option(
    "--fmax", type=float, default=8000, help="Upper frequency boundary for MFCC, in Hz."
)
@click.option("--n-fft", type=int, default=512, help="Number of FFT bins computed.")
@click.option("--n-mels", type=int, default=128, help="Number of Mel bins.")
@click.option(
    "--seed", type=int, default=0, help="Random state seed for dataset partition."
)
def build_decoder_dataset(data_dir, save_dir, **kwargs):

    data_dir = pathlib.Path(data_dir)
    ckpt_dir = pathlib.Path(save_dir)
    dataset = DecoderDataset(data_dir=data_dir, **kwargs).get().checkpoint(ckpt_dir)

    click.echo("Training set size:")
    click.echo(dataset.train_data[0].shape)
    click.echo("Test set size:")
    click.echo(dataset.train_data[1].shape)

    return
