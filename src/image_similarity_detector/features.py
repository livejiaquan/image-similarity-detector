"""Lightweight perceptual fingerprints and an optional batched ResNet50 encoder."""

from __future__ import annotations

import importlib.metadata
import warnings
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps


class Encoder(Protocol):
    dimensions: int
    metric: str
    fingerprint: str

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...

    def metadata(self) -> dict[str, str | int]: ...


def load_image(path: str) -> Image.Image:
    # First frame only for animated/multipage formats; orientation is normalized.
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")


class DHashEncoder:
    dimensions = 256
    metric = "hamming"
    fingerprint = "dhash-v1-16x16-exif-rgb-lanczos"

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        features = []
        for image in images:
            pixels = np.asarray(image.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
            features.append((pixels[:, 1:] > pixels[:, :-1]).reshape(-1).astype(np.float32))
        return np.stack(features)

    def metadata(self) -> dict[str, str | int]:
        return {
            "name": "dhash",
            "metric": self.metric,
            "dimensions": self.dimensions,
            "fingerprint": self.fingerprint,
            "device": "cpu",
        }


class ResNet50Encoder:
    dimensions = 2048
    metric = "cosine"

    def __init__(self, device: str = "auto") -> None:
        try:
            import torch
            from torchvision.models import ResNet50_Weights, resnet50
        except (ImportError, RuntimeError) as error:
            raise ValueError(
                'ResNet50 needs compatible torch/torchvision. Install with pip install ".[neural]".'
            ) from error
        self.torch = torch
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Select --device cpu or auto.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable. Select --device cpu or auto.")
        self.device = device
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.fingerprint = (
            f"resnet50-v2-exif-rgb-torch-{torch.__version__}"
            f"-torchvision-{importlib.metadata.version('torchvision')}-device-{device}"
        )
        self.transform = weights.transforms()
        try:
            model = resnet50(weights=weights)
            model.fc = torch.nn.Identity()
            self.model = model.eval().to(device)
        except Exception as error:
            raise ValueError(
                "Could not initialize ResNet50. First use downloads ImageNet weights; "
                "check connectivity and available device memory."
            ) from error

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.torch.stack([self.transform(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            features = self.model(batch)
        return features.cpu().numpy().astype(np.float32)

    def metadata(self) -> dict[str, str | int]:
        return {
            "name": "resnet50",
            "metric": self.metric,
            "dimensions": self.dimensions,
            "fingerprint": self.fingerprint,
            "device": self.device,
            "weights": "IMAGENET1K_V2",
        }


def create_encoder(backend: str, device: str = "auto") -> Encoder:
    if backend == "dhash":
        return DHashEncoder()
    if backend == "resnet50":
        return ResNet50Encoder(device)
    raise ValueError(f"Unknown backend: {backend}")
