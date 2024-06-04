import pytest
import uuid

from canarygan.decoder import DecoderDataset, KNNDecoder, SVMDecoder, ESNDecoder, decode


@pytest.mark.parametrize("reduce_time_policy",["max","mean","argmax","sum"])
def test_esn(reduce_time_policy, tmp_dummy_data):
    dataset = DecoderDataset(data_dir=tmp_dummy_data).get()
    decoder = ESNDecoder(reduce_time_policy=reduce_time_policy, name=str(uuid.uuid4()))
    decoder.fit(*dataset.train_data)
    y_pred = decoder.predict(dataset.test_data[0])

    assert y_pred.shape == dataset.test_data[1].shape


@pytest.mark.parametrize("decoder", [KNNDecoder(n_neighbors=5), SVMDecoder()])
def test_sklearn_decoders(decoder, tmp_dummy_data):
    dataset = DecoderDataset(data_dir=tmp_dummy_data).get()
    decoder.fit(*dataset.train_data)
    y_pred = decoder.predict(dataset.test_data[0])

    assert y_pred.shape == dataset.test_data[1].shape


@pytest.mark.parametrize("decoder", [ESNDecoder(), KNNDecoder(n_neighbors=5), SVMDecoder()])
def test_decode(decoder, tmp_dummy_data):
    
    dataset = DecoderDataset(data_dir=tmp_dummy_data).get()
    decoder.fit(*dataset.train_data)

    dataset = list(tmp_dummy_data.glob("**/*.wav"))
    r = decode(decoder, dataset)

    assert r.shape == (len(dataset),)

