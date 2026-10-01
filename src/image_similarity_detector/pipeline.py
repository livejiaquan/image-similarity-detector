"""Coordinate discovery, batch feature extraction, matching and scan accounting."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
from PIL import Image

from . import __version__
from .cache import FeatureCache
from .discovery import discover, file_digest
from .features import Encoder, create_encoder, load_image
from .matching import compare, exact_groups
from .models import ImageRecord, ScanConfig, ScanIssue, ScanResult

LOGGER = logging.getLogger(__name__)


def scan(config: ScanConfig, *, encoder: Encoder | None = None) -> ScanResult:
    config.validate()
    start = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    discovered, issues = discover(config.roots)
    if not discovered:
        raise ValueError("No supported images found in the input directories.")
    encoder = encoder or create_encoder(config.backend, config.device)
    cache = (
        FeatureCache(config.cache_dir.expanduser(), encoder.fingerprint, encoder.dimensions)
        if config.cache_dir
        else None
    )
    records: list[ImageRecord] = []
    vectors: list[np.ndarray | None] = []
    pending: list[tuple[int, Image.Image]] = []
    hits = 0

    def flush() -> None:
        if not pending:
            return
        try:
            encoded = encoder.encode([image for _, image in pending])
            if (
                encoded.shape != (len(pending), encoder.dimensions)
                or not np.isfinite(encoded).all()
            ):
                raise ValueError("Encoder returned invalid features; scan stopped.")
            for (index, _), feature in zip(pending, encoded, strict=True):
                vectors[index] = feature
                if cache:
                    cache.put(records[index].sha256, feature)
        finally:
            for _, image in pending:
                image.close()
            pending.clear()

    try:
        for candidate in discovered:
            image: Image.Image | None = None
            try:
                before = candidate.path.stat()
                digest = file_digest(candidate.path)
                image = load_image(str(candidate.path))
                after = candidate.path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("File changed during scan; retry with a stable dataset.")
            except (
                OSError,
                ValueError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as error:
                if image:
                    image.close()
                issues.append(ScanIssue(str(candidate.path), str(error)))
                continue
            index = len(records)
            records.append(
                ImageRecord(
                    index,
                    str(candidate.path),
                    candidate.root,
                    candidate.relative_path,
                    digest,
                    image.width,
                    image.height,
                    after.st_size,
                )
            )
            feature = cache.get(digest) if cache else None
            # A damaged cache must never change the backend's mathematical meaning.
            if feature is not None and encoder.metric == "hamming":
                if not ((feature == 0) | (feature == 1)).all():
                    feature = None
            if feature is not None and encoder.metric == "cosine" and np.linalg.norm(feature) <= 0:
                feature = None
            vectors.append(feature)
            if feature is not None:
                hits += 1
                image.close()
            else:
                pending.append((index, image))
            if len(pending) >= config.batch_size:
                flush()
                LOGGER.info("Encoded %d / %d discovered images", len(records), len(discovered))
        flush()
    finally:
        for _, image in pending:
            image.close()
    if not records:
        raise ValueError("No readable images found. Check file formats and permissions.")
    # Every requested split must contribute readable images. Otherwise an empty
    # train/test directory could silently pass even a strict cross-root gate.
    readable_roots = {record.root for record in records}
    for root in sorted(config.roots, key=lambda root: root.label):
        if root.label not in readable_roots:
            issues.append(
                ScanIssue(
                    str(root.path.resolve()),
                    f"No readable images in input '{root.label}'. "
                    "Check that this directory contains supported, readable image files.",
                )
            )
    features = np.stack(vectors)
    LOGGER.info("Comparing %d images in blocks of %d", len(records), config.block_size)
    matches, counts = compare(
        features,
        records,
        metric=encoder.metric,
        threshold=config.threshold,
        scope=config.scope,
        block_size=config.block_size,
        max_pairs=config.max_pairs,
    )
    groups = exact_groups(records, config.scope)
    warnings = [
        "Near matches are review candidates, not deletion instructions.",
        "Cross-root matches may indicate split contamination; validate dataset provenance.",
        "Animated and multipage files are compared using their first frame only.",
    ]
    if counts["matching_pairs"] > len(matches):
        warnings.append(
            f"Match storage capped at {config.max_pairs}; totals cover all comparisons. "
            "Stored pairs follow deterministic block traversal and are not a top-score ranking."
        )
    return ScanResult(
        schema_version="1.0",
        tool_version=__version__,
        started_at=started_at,
        duration_seconds=round(perf_counter() - start, 3),
        config=config.to_dict(),
        backend=encoder.metadata(),
        images=records,
        matches=matches,
        exact_groups=groups,
        issues=issues,
        summary={
            "status": "partial" if issues else "complete",
            "images_discovered": len(discovered),
            "images_scanned": len(records),
            "issues_count": len(issues),
            "cache_hits": hits,
            **counts,
            "pairs_stored": len(matches),
            "pairs_truncated": counts["matching_pairs"] > len(matches),
            "exact_groups": len(groups),
        },
        warnings=warnings,
    )
