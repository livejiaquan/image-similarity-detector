"""Typed configuration and versioned, serializable scan results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputRoot:
    label: str
    path: Path


@dataclass(frozen=True)
class ScanConfig:
    roots: tuple[InputRoot, ...]
    backend: str = "dhash"
    threshold: float = 0.95
    scope: str = "all"
    batch_size: int = 32
    block_size: int = 256
    max_pairs: int = 10_000
    report_pairs: int = 100
    device: str = "auto"
    cache_dir: Path | None = None

    def validate(self) -> None:
        if not self.roots:
            raise ValueError("At least one input directory is required.")
        if self.backend not in {"dhash", "resnet50"}:
            raise ValueError("Backend must be dhash or resnet50.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        if self.scope not in {"all", "cross-root"}:
            raise ValueError("Scope must be all or cross-root.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Device must be auto, cpu, cuda or mps.")
        if min(self.batch_size, self.block_size, self.max_pairs, self.report_pairs) < 1:
            raise ValueError("Batch size, block size and report limits must be positive.")
        if self.scope == "cross-root" and len(self.roots) < 2:
            raise ValueError("Cross-root comparison requires at least two input directories.")
        labels = [r.label for r in self.roots]
        if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
            raise ValueError("Input labels must be non-empty and unique.")
        for i, root in enumerate(self.roots):
            if root.path.is_symlink():
                raise ValueError(f"Input root may not be a symbolic link: {root.path}")
            if not root.path.is_dir():
                raise ValueError(f"Input directory does not exist: {root.path}")
            resolved = root.path.resolve()
            for other in self.roots[:i]:
                other_path = other.path.resolve()
                if (
                    resolved == other_path
                    or resolved in other_path.parents
                    or other_path in resolved.parents
                ):
                    raise ValueError("Input roots must not overlap; this would count images twice.")
            if self.cache_dir:
                cache_path = self.cache_dir.expanduser().resolve()
                if cache_path == resolved or resolved in cache_path.parents:
                    raise ValueError("Cache directory must be outside all input roots.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [{"label": r.label, "path": str(r.path.resolve())} for r in self.roots],
            "backend": self.backend,
            "threshold": self.threshold,
            "scope": self.scope,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "max_pairs": self.max_pairs,
            "report_pairs": self.report_pairs,
            "device": self.device,
            "cache_enabled": self.cache_dir is not None,
        }


@dataclass(frozen=True)
class ImageRecord:
    id: int
    path: str
    root: str
    relative_path: str
    sha256: str
    width: int
    height: int
    size_bytes: int


@dataclass(frozen=True)
class ScanIssue:
    path: str
    reason: str


@dataclass(frozen=True)
class Match:
    left: int
    right: int
    similarity: float
    kind: str
    cross_root: bool


@dataclass
class ScanResult:
    schema_version: str
    tool_version: str
    started_at: str
    duration_seconds: float
    config: dict[str, Any]
    backend: dict[str, Any]
    images: list[ImageRecord]
    matches: list[Match]
    exact_groups: list[list[int]]
    issues: list[ScanIssue]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
