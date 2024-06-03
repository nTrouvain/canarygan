# Author: Nathan Trouvain at 9/15/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import uuid

import torch
import scipy.io.wavfile as wav

from tqdm import tqdm

from pmseq.canarygan.train_lightning import CanaryGAN


if __name__ == "__main__":

    sr = 16000

    gan = CanaryGAN.load_from_checkpoint("pmseq/canarygan/checkpoints/all/last.ckpt")
    generator = gan.generator
    generator.eval()

    z = torch.rand(10000, 3).cuda()

    waves = generator(z).cpu().detach().numpy()

    output_dir = pathlib.Path("data/generated-scratch/")
    output_dir.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4()

    for i, wave in enumerate(tqdm(waves)):

        filename = f"gen-{i}_{uid}.wav"

        wav.write(
            output_dir / filename,
            sr,
            wave.flatten()[:sr],
        )
