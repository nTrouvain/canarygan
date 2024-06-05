import pytest
import numpy as np

from canarygan.gan import generate, generate_and_decode, CanaryGANGenerator
from canarygan.decoder import DecoderDataset, KNNDecoder
from canarygan.const import SliceLengths


@pytest.mark.parametrize(
    "slice_len", (SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG)
)
def test_generate(slice_len):
    generator = CanaryGANGenerator(slice_len=slice_len)
    z = np.random.uniform(-1, 1, (100, 3))
    x = generate(generator, z)

    assert x.shape == (100, slice_len.value)


@pytest.mark.parametrize(
    "z", (np.random.uniform(-1, 1, (10, 3)), np.random.uniform(-1, 1, (10, 1, 3)))
)
def test_generate_and_decode(z, tmp_dummy_data):
    dataset = DecoderDataset(data_dir=tmp_dummy_data).get()
    decoder = KNNDecoder(n_neighbors=5).fit(*dataset.train_data)
    generator = CanaryGANGenerator()

    x, y, z_ = generate_and_decode(decoder, generator, z, save_gen=True)

    assert x.shape == (10, SliceLengths.SHORT.value)
    assert y.shape == (10,)
    assert z_.shape == (10, 3)
