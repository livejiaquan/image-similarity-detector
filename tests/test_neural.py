"""No-download architecture smoke test; actual pretrained weights are tested separately."""

import numpy as np
import pytest
from PIL import Image


def test_resnet_batch_output_with_local_random_weights(monkeypatch):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision.models")
    from image_similarity_detector.features import ResNet50Encoder

    original = torchvision.resnet50
    monkeypatch.setattr(torchvision, "resnet50", lambda weights: original(weights=None))
    encoder = ResNet50Encoder("cpu")
    images = [Image.new("RGB", (48, 36), "red"), Image.new("RGB", (36, 48), "blue")]
    torch.set_num_threads(2)
    encoded = encoder.encode(images)
    assert encoded.shape == (2, 2048)
    assert np.isfinite(encoded).all()
    assert not np.array_equal(encoded[0], encoded[1])
