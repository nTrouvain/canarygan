from tempfile import TemporaryDirectory
from pathlib import Path

import pytest
import numpy as np
from scipy.io import wavfile

from canarygan.const import SliceLengths


@pytest.fixture(scope="session")
def tmp_dummy_data():

    with TemporaryDirectory() as tmpdir:

        root = Path(tmpdir)
        classes = ["A", "B", "C"]
        for c in classes:
            subdir = root / c
            subdir.mkdir(exist_ok=True)

            for i in range(16):
                a = np.random.uniform(-1, 1, SliceLengths.SHORT.value)
                wavfile.write(subdir / f"{i}.wav", 16000, a)

        yield root


@pytest.fixture(scope="function")
def tmp_save_dir():
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
