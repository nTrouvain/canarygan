import pytest
import torch

from canarygan.inception import (
    SyllableClassifier,
    train,
)
from canarygan.const import SliceLengths


@pytest.mark.parametrize(
    "slice_len", (SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG)
)
def test_model(slice_len):

    clf = SyllableClassifier(
        n_classes=16,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=128,
        f_min=40,
        f_max=7800,
        n_mels=128,
        pad_mode="constant",
        center=True,
        slice_length=slice_len,
        n_channels=128,
    )

    x = torch.rand((10, 1, slice_len.value)).to("cpu:0")
    out = clf(x)
    out = out.detach().numpy()
    assert out.shape == (10, 16)


def test_training(tmp_dummy_data, tmp_save_dir):
    train(
        save_dir=tmp_save_dir / "inception",
        data_dir=tmp_dummy_data,
        max_epochs=1,
        batch_size=16,
        devices=1,
        num_nodes=1,
        num_workers=1,
        log_every_n_steps=1,
        save_every_n_epochs=1,
        save_topk=1,
        seed=4862,
        resume=False,
        dry_run=False,
        early_stopping=False,
    )

    assert (tmp_save_dir / "inception" / "checkpoints").exists()
    assert (tmp_save_dir / "inception" / "logs").exists()

