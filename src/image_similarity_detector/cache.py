"""Content-addressed feature cache with atomic writes and shape validation."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class FeatureCache:
    def __init__(self, directory: Path, fingerprint: str, dimensions: int) -> None:
        namespace = hashlib.sha256(fingerprint.encode()).hexdigest()[:20]
        self.directory = directory / namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions

    def get(self, digest: str) -> np.ndarray | None:
        path = self.directory / f"{digest}.npy"
        try:
            feature = np.load(path, allow_pickle=False)
            if not isinstance(feature, np.ndarray):
                feature.close()
                return None
            if feature.dtype.kind not in "buif":
                return None
            if feature.shape != (self.dimensions,) or not np.isfinite(feature).all():
                return None
            return feature.astype(np.float32)
        except (OSError, ValueError, EOFError):
            return None

    def put(self, digest: str, feature: np.ndarray) -> None:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, suffix=".tmp", delete=False
            ) as stream:
                temporary = stream.name
                np.save(stream, feature, allow_pickle=False)
            os.replace(temporary, self.directory / f"{digest}.npy")
        except OSError as error:
            LOGGER.warning("Could not write feature cache: %s", error)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
