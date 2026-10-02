from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dataset(tmp_path: Path):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    pixels = np.random.default_rng(42).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(train / "original.png")
    (test / "copy.png").write_bytes((train / "original.png").read_bytes())
    Image.fromarray(pixels).save(test / "recompressed.jpg", quality=94)
    Image.fromarray(255 - pixels).save(train / "different.png")
    return train, test
