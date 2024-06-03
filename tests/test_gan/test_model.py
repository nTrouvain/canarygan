import pytest
import torch

from canarygan.gan.model import CanaryGANDiscriminator, CanaryGANGenerator, CanaryGAN
from canarygan.gan.params import SliceLengths


def test_discriminator():
    for slice_len in [SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG]:
        discriminator = CanaryGANDiscriminator(
            dim=64,
            nch=1,
            kernel_len=25,
            stride=4,
            phasesuffle_rad=8,
            use_batchnorm=True,
            slice_len=slice_len,
        )

        x = torch.rand((1, 1, slice_len.value))
        out = discriminator(x)
        assert out.shape[-1] == 1


def test_generator():
    for slice_len in [SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG]:
        for upsample in ["nn", "zeros"]:

            generator = CanaryGANGenerator(
                dim=64,
                latent_dim=3,
                nch=1,
                kernel_len=25,
                stride=4,
                padding="same",
                upsample=upsample,
                use_batchnorm=False,
                slice_len=slice_len,
            )

            z = torch.Tensor([[0.1, 0.2, 0.1]])

            out = generator(z)
            assert out.shape[-1] == slice_len.value
