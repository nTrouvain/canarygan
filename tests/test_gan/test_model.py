import pytest
import torch

from canarygan.gan import CanaryGANDiscriminator, CanaryGANGenerator
from canarygan.const import SliceLengths


@pytest.mark.parametrize(
    "slice_len", (SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG)
)
def test_discriminator(slice_len):
    discriminator = CanaryGANDiscriminator(
        dim=64,
        nch=1,
        kernel_len=25,
        stride=4,
        phasesuffle_rad=8,
        use_batchnorm=True,
        slice_len=slice_len,
    ).to("cpu:0")

    x = torch.rand((10, 1, slice_len.value)).to("cpu:0")
    out = discriminator(x)
    out = out.detach().numpy()
    assert out.shape == (10,)


@pytest.mark.parametrize("upsample", ("nn", "zeros"))
@pytest.mark.parametrize(
    "slice_len", (SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG)
)
def test_generator(upsample, slice_len):
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
    ).to("cpu:0")

    z = torch.Tensor([[0.1, 0.2, 0.1]]).to("cpu:0")

    out = generator(z)
    out = out.detach().numpy()
    assert out.shape[-1] == slice_len.value
