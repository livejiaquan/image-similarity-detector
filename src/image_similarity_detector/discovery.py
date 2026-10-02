"""Deterministic discovery. Input directories and symlinks are never modified."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .models import InputRoot, ScanIssue

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DiscoveredImage:
    path: Path
    root: str
    relative_path: str


def discover(roots: tuple[InputRoot, ...]) -> tuple[list[DiscoveredImage], list[ScanIssue]]:
    images: list[DiscoveredImage] = []
    issues: list[ScanIssue] = []
    for root in roots:
        root_path = root.path.resolve()

        def onerror(error: OSError) -> None:
            issues.append(ScanIssue(str(error.filename), f"Directory unreadable: {error.strerror}"))

        for directory, dirs, files in os.walk(root_path, followlinks=False, onerror=onerror):
            traversable = []
            for name in sorted(dirs):
                child = Path(directory) / name
                if child.is_symlink():
                    issues.append(ScanIssue(str(child), "Symbolic link directory skipped."))
                else:
                    traversable.append(name)
            dirs[:] = traversable
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                if path.is_symlink():
                    issues.append(ScanIssue(str(path), "Symbolic link skipped."))
                    continue
                images.append(
                    DiscoveredImage(path, root.label, path.relative_to(root_path).as_posix())
                )
    images.sort(key=lambda image: (image.root, image.relative_path))
    return images, issues


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
