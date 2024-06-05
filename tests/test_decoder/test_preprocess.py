import pytest

from canarygan.decoder import preprocess, DecoderDataset


@pytest.mark.parametrize("feats", ["deltas", "spec", "mel", "mfcc-classic"])
def test_preprocess(tmp_dummy_data, feats):

    dataset = tmp_dummy_data.glob("**/*.wav")

    r = preprocess(
        dataset,
        features=feats,
        n_fft=1024,
        win_length=256,
        hop_length=128,
        fmin=500,
        fmax=8000,
        n_mels=128,
        n_mfcc=13,
        mfcc=False,
        padding="interp",
        lifter=0,
        sampling_rate=16000,
    )

    assert isinstance(r, list)
    assert r[0].ndim == 2


def test_dataset(tmp_dummy_data, tmp_save_dir):

    dataset_ckpt = tmp_save_dir / "decoder-dataset"
    dataset = DecoderDataset(data_dir=tmp_dummy_data, checkpoint_dir=dataset_ckpt)
    dataset = dataset.get().checkpoint()

    dataset = DecoderDataset.from_checkpoint(dataset_ckpt)

    assert dataset.train_data != None
    assert dataset.test_data != None
    assert dataset.class_labels == ["A", "B", "C"]


def test_dataset_params(tmp_save_dir):
    param_file = tmp_save_dir / "decoder-params.yml"
    dataset = DecoderDataset()
    dataset.transform_kwargs["sampling_rate"] = 50000
    dataset.save_params(param_file)

    dataset = DecoderDataset()
    dataset.load_params(param_file)

    assert dataset.transform_kwargs["sampling_rate"] == 50000
