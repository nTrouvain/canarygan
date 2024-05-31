# Author: Nathan Trouvain at 9/15/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import argparse

import numpy as np

from pmseq.inverse_model import train_inverse_model
from pmseq.utils import load_generator, load_decoder, load_latent_vects, select_device


parser = argparse.ArgumentParser()

parser.add_argument("generator_ckpt", type=str)
parser.add_argument("decoder_ckpt", type=str)
parser.add_argument("-o", "--save_file", type=str)
parser.add_argument("--vec_file", type=str, default=None)
parser.add_argument("--n_samples", type=int, default=16000)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--eta", type=float, default=0.01)
parser.add_argument("--w_init_high", type=float, default=0.001)
parser.add_argument("--w_init_low", type=float, default=-0.001)
parser.add_argument("--activation", type=str, choices=["max", "mean"], default="max")
parser.add_argument("--eval_every_n_steps", type=int, default=15)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--device_idx", type=int, default=0)


def main(
    generator_ckpt,
    decoder_ckpt,
    save_file,
    vec_file,
    n_samples,
    max_steps,
    eta,
    w_init_high,
    w_init_low,
    activation,
    eval_every_n_steps,
    device,
    device_idx,
):
    
    device = select_device(device, device_idx)
    generator = load_generator(generator_ckpt, device=device)
    decoder = load_decoder(decoder_ckpt)
    
    motor_vects = load_latent_vects(vec_file, n_samples=n_samples)
   
    last_w, w_history, m_base_register, x_gens, y_ms, class_to_idx, p95 = train_inverse_model(
        decoder,
        generator,
        motor_vects,
        max_steps=max_steps,
        eta=eta,
        w_dist={"low": w_init_low, "high": w_init_high},
        activation=activation,
        device=device,
        eval_every_n_steps=eval_every_n_steps,
    )
    
    save_file = pathlib.Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        save_file,
        last_w=last_w,
        w_history=w_history,
        m_base_register=m_base_register,
        x_gens=x_gens,
        y_ms=y_ms,
        p95=p95,
    )
    np.savez(
        save_file.with_suffix(".class_to_idx"),
        **class_to_idx,
    )

    print(f"Saved results at {save_file}")


if __name__ == "__main__":

    args = parser.parse_args()
    main(**vars(args))

