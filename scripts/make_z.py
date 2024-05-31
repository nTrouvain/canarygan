import pathlib
import argparse

import numpy as np


def main(
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

    print(f"Generating array of size {size}.")
    z = rvs(**dist_params, size=size)

    save_dir = pathlib.Path(save_dir)

    (save_dir / f"latent_z-dim_{dim}.npy").unlink(missing_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"latent_z-dim_{dim}.npy", z)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sample GAN latent space and save vectors to disk.")
    
    parser.add_argument("save_dir", type=str, help="Directory where vectors will be stored.")
    parser.add_argument("n_samples", type=int, help="Number of samples to draw.")
    parser.add_argument("-d", "--dim", type=int, default=3, help="Latent space dimension.")
    parser.add_argument("--seed", type=int, default=0, help="Random state seed.")

    args = parser.parse_args()

    main(**vars(args))

    print("Done!")
