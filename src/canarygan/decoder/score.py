try:
    import dill as pickle
except ImportError:
    print("'dill' not found. Falling back on pickle.")
    import pickle
from pathlib import Path
from collections import Counter

import yaml
import torch
import numpy as np
import pandas as pd
import reservoirpy as rpy

from joblib import delayed, Parallel
from scipy.special import softmax, log_softmax
from torch.utils import data
from rich.progress import track

from ..canarygan.generator import CanaryGANGenerator
from ..canarygan.dataset import LatentSpaceSamples
from ..decoder.preprocessing import mfcc
from ..decoder.dataset import DecoderDataset


def is_score(logits, n_split=10):
    groups = np.split(logits, n_split)
    scores = np.zeros(n_split)

    for i in range(n_split):
        y = groups[i]
        p = softmax(y, axis=1)
        mean_p = np.mean(p, axis=0, keepdims=True)
        logp = log_softmax(y, axis=1)
        kl = p * (logp - np.log(mean_p))
        kl = np.mean(np.sum(kl, axis=1))
        scores[i] = np.exp(kl)

    return np.mean(scores), np.std(scores)


@delayed
def preprocess_sound(y, sampling_rate, hop_length, win_length, lifter, n_mfcc, padding):
    features = mfcc(y, sampling_rate, hop_length, win_length, lifter, n_mfcc, padding)
    return np.concatenate(features[1:], axis=1)


def decode(
    decoder,
    generator,
    latent_vects,
    device,
    save_in=None,
):
    # Used to get preprocessing parameters
    d = DecoderDataset()

    generator.eval()
    y_gens = []
    x_gens = []
    with torch.no_grad():
        for z in track(latent_vects):
            z = z.cuda(device)
            x_gen = generator(z)

            x_gen = x_gen.squeeze().cpu().numpy()

            if save_in is not None:
                x_gens.append(x_gen)

            with Parallel(n_jobs=-1) as parallel:
                x_mfcc = parallel(
                    [
                        preprocess_sound(
                            x,
                            d.sampling_rate,
                            d.hop_length,
                            d.win_length,
                            d.lifter,
                            d.n_mfcc,
                            d.padding,
                        )
                        for x in x_gen
                    ]
                )

            y_gen = decoder.run(x_mfcc)

            if isinstance(y_gen, list):
                y_gen = np.r_[y_gen]
            
            y_gens.append(y_gen)

    y_gens = np.concatenate(y_gens, axis=0)

    if save_in is not None:
        save_dir = Path(save_in)
        if save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir
        else:
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            filename = "generation"

        x_gens = np.concatenate(x_gens, axis=0)

        np.save(save_dir / f"{filename}_audio.npy", x_gens)
        np.save(save_dir / f"{filename}_labels.npy", y_gens)

    return y_gens


def score(
    vec_file,
    save_dir,
    generator_version,
    decoder_ckpt,
    n_is_split=10,
    batch_size=100,
    num_workers=8,
    device_idx=0,
):
    rpy.verbosity(0)

    dataset = LatentSpaceSamples(vec_file)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if torch.cuda.device_count() > 1:
        device = device_idx
    elif torch.cuda.is_available():
        device = 0
    else:
        raise RuntimeError("No GPU available.")

    with open(decoder_ckpt, "rb") as fp:
        decoder = pickle.load(fp)

    idx_to_class = {v: k for k, v in decoder.class_to_idx.items()}

    all_ckpts = sorted((Path(generator_version) / "all").rglob("*.ckpt"))

    if len(all_ckpts) == 0:
        print(f"No ckpt file found at {generator_version}/all.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    version = Path(generator_version).stem.split("_")[-1]

    all_scores = []
    for generator_ckpt in all_ckpts:
        epoch = int(generator_ckpt.stem.split("-")[0].split("_")[-1])

        generator_ckpt = Path(generator_ckpt)
        ckpt = torch.load(
            generator_ckpt, map_location=lambda storage, loc: storage.cuda(device)
        )
        state_dict = {
            ".".join(k.split(".")[1:]): v
            for k, v in ckpt["state_dict"].items()
            if "generator" in k
        }
        generator = CanaryGANGenerator().cuda(device)

        try:
            status = generator.load_state_dict(state_dict)
            print(f"Ckpt load status: {str(status)}")
        except Exception as e:
            print(e)
            continue

        # Unormalized linear predictions (no softmax), all timesteps
        logits = decode(decoder, generator, dataloader, device=device)

        # Reduce time dimension
        logits = np.sum(logits, axis=1)

        # Inception score
        is_mean, is_std = is_score(logits, n_split=n_is_split)
        print(f"IS - epoch {epoch}: {is_mean}±{is_std}")

        # Compute labels
        y_pred = np.argmax(softmax(logits, axis=1), axis=1)
        y_pred = [idx_to_class[y] for y in y_pred]

        class_counts = Counter(y_pred)

        print(class_counts)

        scores = {
            "epoch": epoch,
            "is_mean": is_mean,
            "is_std": is_std,
        }
        scores.update(class_counts)

        with open(save_dir / f"part-decoder_gan_scores-version_{version}-epoch_{epoch}.yml", "w+") as fp:
            yaml.dump(scores, fp, Dumper=yaml.Dumper) 

        all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    scores_df.to_csv(save_dir / f"decoder_gan_scores-version_{version}.csv")
