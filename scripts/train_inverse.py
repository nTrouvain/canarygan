# Author: Nathan Trouvain at 9/15/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from functools import partial
import re
import argparse

from pathlib import Path

import numpy as np
import librosa as lbr
import matplotlib.pyplot as plt

from pmseq.inverse_model import train_inverse_random_and_stop, create_learning_set, act_and_decode
from pmseq.utils import load_generator, load_decoder, load_latent_vects, select_device


parser = argparse.ArgumentParser()

parser.add_argument("generator_ckpt", type=str)
parser.add_argument("decoder_ckpt", type=str)
parser.add_argument("--epoch", type=int, default=None)
parser.add_argument("-o", "--save_dir", type=str)
parser.add_argument("--vec_file", type=str, default=None)
parser.add_argument("--learning_set", type=str, default=None)
parser.add_argument("--n_samples", type=int, default=16000)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--n_instances", type=int, default=3)
parser.add_argument("--eta", type=float, default=0.01)
parser.add_argument("--kind", type=str, default="none")
parser.add_argument("--margin", type=float, default=0.01)
parser.add_argument("--tau", type=float, default=10)
parser.add_argument("--w_init_high", type=float, default=0.001)
parser.add_argument("--w_init_low", type=float, default=-0.001)
parser.add_argument("--activation", type=str, choices=["max", "mean"], default="max")
parser.add_argument("--eval_every_n_steps", type=int, default=15)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--device_idx", type=int, default=0)


def mean_spec_plot(
        y_gens,
        x_gens,
        idx_to_class,
        save_file,
        y_threshold=0.99,
        sr=16000,
        version=None, 
        epoch=None,
    ):
    fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True, sharey=True)
    axs = axs.flatten()

    fig.suptitle(f"version={version}, epoch={epoch}")
    
    for i in range(16):
        y = np.concatenate(y_gens[:, i, :], axis=0)
        x = np.concatenate(x_gens[:, i, :, :round(sr*0.3)], axis=0)
        x_sel = x[y >= y_threshold]
        n = x_sel.shape[0]
        ax = axs[i]
        if x_sel.shape[0] > 0:
            # from Silvia's code
            s = lbr.stft(y=x_sel, n_fft=256, hop_length=64)
            s = np.log(1 + 100 * np.abs(s)**2)
            sm = np.mean(s, axis=0)
            
            ax.imshow(sm, origin="lower", aspect="auto", cmap="inferno")

        lbl = idx_to_class[i]
        ax.set_title(f"{lbl} (n={n})")

    fig.savefig(save_file, bbox_inches="tight")
    plt.close()
    return


def train_all(
    generator_ckpt,
    decoder_ckpt,
    epoch,
    save_dir,
    vec_file,
    learning_set,
    n_samples,
    max_steps,
    n_instances,
    eta,
    margin,
    kind,
    tau,
    w_init_high,
    w_init_low,
    activation,
    eval_every_n_steps,
    device,
    device_idx,
):
    device = select_device(device, device_idx)
    decoder = load_decoder(decoder_ckpt)
    ckpts = Path(generator_ckpt)
    motor_vects = load_latent_vects(vec_file, n_samples=n_samples)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    has_data = True
    if learning_set is not None:
        learning_set = Path(learning_set) / f"activation_{activation}"
        learning_set.mkdir(parents=True, exist_ok=True)
        if len(list(learning_set.glob("*"))) == 0:
            has_data = False
    else:
        has_data = False

    for ckpt in sorted(ckpts.glob(f"**/version_*/all/*epoch_{epoch}*.ckpt")):
        print(ckpt)
        m = re.search(r".*version_(?P<version>[0-9]+).*all_epoch_.*ckpt", str(ckpt))
        (version,) = m.groups()
        print(version)
        generator = load_generator(ckpt, device=device)
       
        if has_data:
            set_file = learning_set / f"learning_set-version_{version}-epoch_{epoch}.npz"

            d = np.load(set_file)
            p_rs = d["p_rs"]
            m_rs = d["m_rs"]
            p95 = d["p95"]
            classes = d["sorted_classes"]
            class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            p_rs, m_rs, p95, class_to_idx = create_learning_set(
                motor_vects,
                decoder,
                generator,
                activation,
                device,
            )

            if learning_set is not None:
                np.savez(
                    learning_set / f"learning_set-version_{version}-epoch_{epoch}.npz",
                    p_rs=p_rs,
                    m_rs=m_rs,
                    p95=p95,
                    sorted_classes=np.array(sorted([c for c in class_to_idx.keys()], key=lambda x: class_to_idx[x])),
                )

        np.random.seed(0)
        
        actor_fn = partial(
            act_and_decode, 
            decoder=decoder,
            generator=generator, 
            activation=activation, 
            device=device
        )

        store = {}
        for i in range(n_instances):
            last_w, w_history, xis, reaches, criterions, y_gens, x_gens, m_r_base = train_inverse_random_and_stop(
                p_rs, 
                m_rs, 
                p95=p95,
                w_dist={"low": w_init_low, "high": w_init_high},
                actor_fn=actor_fn,
                eta=eta,
                margin=margin,
                kind=kind,
                tau=tau,
                max_steps=max_steps,
                eval_every_n_steps=eval_every_n_steps,
            )
            
            store[i] = dict(
                last_w=last_w,
                w_history=w_history,
                xi_history=xis,
                m_r_base=m_r_base,
                x_gens=x_gens,
                y_gens=y_gens,
                reaches=reaches,
                criterions=criterions,
            )

        x_gens = np.vstack([store[i]["x_gens"][np.newaxis] for i in range(n_instances)])

        store = dict(
            last_w=np.vstack([store[i]["last_w"][np.newaxis] for i in range(n_instances)]),
            w_history=np.vstack([store[i]["w_history"][np.newaxis] for i in range(n_instances)]),
            xi_history=np.vstack([store[i]["xi_history"][np.newaxis] for i in range(n_instances)]),
            m_r_base=np.vstack([store[i]["m_r_base"][np.newaxis] for i in range(n_instances)]),
            y_gens=np.vstack([store[i]["y_gens"][np.newaxis] for i in range(n_instances)]),
            reaches=np.vstack([store[i]["reaches"][np.newaxis] for i in range(n_instances)]),
            criterions=np.vstack([store[i]["criterions"][np.newaxis] for i in range(n_instances)]),
            version=version,
            epoch=epoch,
            tau=tau,
            margin=margin,
            kind=kind,
            decoder_version=Path(decoder_ckpt).stem,
        )

        store["version"] = version
        store["epoch"] = epoch
        store["p95"] = p95
        store["sorted_classes"] = np.array(sorted([c for c in class_to_idx.keys()], key=lambda x: class_to_idx[x]))

        filename = f"kind_{kind}-margin_{margin}-tau_{tau}-version_{version}-epoch_{epoch}"

        mean_spec_plot(
            store["y_gens"],
            x_gens,
            idx_to_class={v:k for k,v in class_to_idx.items()},
            save_file=save_dir / (filename + ".pdf"),
            y_threshold=0.99,
            version=version,
            epoch=epoch,
        )

        save_file = save_dir / (filename + ".npz")
        np.savez(
            save_file,
            **store,
        )


def main(
    generator_ckpt,
    decoder_ckpt,
    save_dir,
    vec_file,
    learning_set,
    n_samples,
    max_steps,
    n_instances,
    eta,
    margin,
    kind,
    tau,
    w_init_high,
    w_init_low,
    activation,
    eval_every_n_steps,
    device,
    device_idx,
    **kwargs,
):  
    np.random.seed(0)

    m = re.search(r".*version_(?P<version>[0-9]+).*all_epoch_(?P<epoch>[0-9]+).*ckpt", str(generator_ckpt))
    md = re.search(r".*decoder-(?P<version>[0-9]+).pkl", str(decoder_ckpt))
    if m is None:
        version = -1
        epoch = -1
    else:
        version, epoch = m.groups()
        epoch = int(epoch)
        version = int(version)

    if md is None:
        decoder_version = -1
    else:
        decoder_version, = md.groups()
        decoder_version = int(decoder_version)
    
    print(f"Version: {version} - Epoch: {epoch}")

    device = select_device(device, device_idx)
    decoder = load_decoder(decoder_ckpt)
    generator = load_generator(generator_ckpt, device=device)
    motor_vects = load_latent_vects(vec_file, n_samples=n_samples)
   
    # Maybe generate dataset of motor vectors/percepts pairs, or load it
    data_file = Path(learning_set)\
        / f"activation_{activation}"\
        / f"learning_set-version_{version}-epoch_{epoch}-decoder_{decoder_version}.npz"

    if Path(data_file).is_file():
        d = np.load(data_file)
        p_rs = d["p_rs"]
        m_rs = d["m_rs"]
        p95 = d["p95"]
        classes = d["sorted_classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
    else:
        data_file.parent.mkdir(parents=True, exist_ok=True)
        p_rs, m_rs, p95, class_to_idx = create_learning_set(
            motor_vects,
            decoder,
            generator,
            activation,
            device,
        )

        if learning_set is not None:
            np.savez(
                data_file,
                p_rs=p_rs,
                m_rs=m_rs,
                p95=p95,
                sorted_classes=np.array(sorted([c for c in class_to_idx.keys()], key=lambda x: class_to_idx[x])),
            )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    actor_fn = partial(
        act_and_decode, 
        decoder=decoder,
        generator=generator, 
        activation=activation, 
        device=device
    )

    store = {}
    for i in range(n_instances):
        last_w, w_history, xis, reaches, criterions, y_gens, x_gens, m_r_base = train_inverse_random_and_stop(
            p_rs, 
            m_rs, 
            p95=p95,
            w_dist={"low": w_init_low, "high": w_init_high},
            actor_fn=actor_fn,
            eta=eta,
            margin=margin,
            kind=kind,
            tau=tau,
            max_steps=max_steps,
            eval_every_n_steps=eval_every_n_steps,
        )
        
        store[i] = dict(
            last_w=last_w,
            w_history=w_history,
            xi_history=xis,
            m_r_base=m_r_base,
            x_gens=x_gens,
            y_gens=y_gens,
            reaches=reaches,
            criterions=criterions,
        )

    x_gens = np.vstack([store[i]["x_gens"][np.newaxis] for i in range(n_instances)])

    store = dict(
        last_w=np.vstack([store[i]["last_w"][np.newaxis] for i in range(n_instances)]),
        w_history=np.vstack([store[i]["w_history"][np.newaxis] for i in range(n_instances)]),
        xi_history=np.vstack([store[i]["xi_history"][np.newaxis] for i in range(n_instances)]),
        m_r_base=np.vstack([store[i]["m_r_base"][np.newaxis] for i in range(n_instances)]),
        y_gens=np.vstack([store[i]["y_gens"][np.newaxis] for i in range(n_instances)]),
        reaches=np.vstack([store[i]["reaches"][np.newaxis] for i in range(n_instances)]),
        criterions=np.vstack([store[i]["criterions"][np.newaxis] for i in range(n_instances)]),
        version=version,
        epoch=epoch,
        tau=tau,
        margin=margin,
        kind=kind,
        decoder_version=decoder_version,
    )

    store["version"] = version
    store["epoch"] = epoch
    store["p95"] = p95
    store["sorted_classes"] = np.array(sorted([c for c in class_to_idx.keys()], key=lambda x: class_to_idx[x]))

    filename = f"kind_{kind}-margin_{margin}-tau_{tau}-version_{version}-epoch_{epoch}-decoder_version_{decoder_version}"

    mean_spec_plot(
        store["y_gens"],
        x_gens,
        idx_to_class={v:k for k,v in class_to_idx.items()},
        save_file=save_dir / (filename + ".pdf"),
        y_threshold=0.99,
        version=version,
        epoch=epoch,
    )

    save_file = save_dir / (filename + ".npz")
    np.savez(
        save_file,
        **store,
    )


if __name__ == "__main__":

    args = parser.parse_args()
    main(**vars(args))

