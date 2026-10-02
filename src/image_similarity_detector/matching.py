"""Exhaustive block comparison; never materializes the full N x N matrix."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .models import ImageRecord, Match


def compare(
    features: np.ndarray,
    records: list[ImageRecord],
    *,
    metric: str,
    threshold: float,
    scope: str,
    block_size: int,
    max_pairs: int,
) -> tuple[list[Match], dict[str, int]]:
    if block_size < 1 or max_pairs < 1 or not 0 <= threshold <= 1:
        raise ValueError("Block size and pair limit must be positive; threshold must be 0–1.")
    if metric not in {"cosine", "hamming"}:
        raise ValueError(f"Unsupported metric: {metric}")
    if (
        features.ndim != 2
        or features.shape[1] < 1
        or len(features) != len(records)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Features must be a finite matrix aligned with image records.")
    vectors = features.astype(np.float32, copy=False)
    if metric == "cosine":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Cosine similarity is undefined for zero feature vectors.")
        vectors = vectors / norms
    elif not ((vectors == 0) | (vectors == 1)).all():
        raise ValueError("Hamming features must contain only binary values.")
    labels = np.array([r.root for r in records])
    # Integer digest codes keep equality checks cheap and independent of floating point.
    digest_codes = {digest: k for k, digest in enumerate(dict.fromkeys(r.sha256 for r in records))}
    digests = np.array([digest_codes[r.sha256] for r in records])
    matches: list[Match] = []
    counts = {
        "pairs_compared": 0,
        "matching_pairs": 0,
        "exact_pairs": 0,
        "near_pairs": 0,
        "cross_root_matching_pairs": 0,
    }
    for i in range(0, len(records), block_size):
        left = vectors[i : i + block_size]
        for j in range(i, len(records), block_size):
            right = vectors[j : j + block_size]
            dot = left @ right.T
            if metric == "cosine":
                scores = np.clip(dot, -1, 1)
            else:
                distances = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - 2 * dot
                scores = np.clip(1 - distances / vectors.shape[1], 0, 1)
            eligible = np.ones(scores.shape, dtype=bool)
            if i == j:
                eligible = np.triu(eligible, k=1)
            cross = labels[i : i + len(left), None] != labels[None, j : j + len(right)]
            exact = digests[i : i + len(left), None] == digests[None, j : j + len(right)]
            scores[exact] = 1.0
            if scope == "cross-root":
                eligible &= cross
            counts["pairs_compared"] += int(eligible.sum())
            candidates = eligible & (scores >= threshold)
            matched_count = int(candidates.sum())
            exact_count = int((candidates & exact).sum())
            counts["matching_pairs"] += matched_count
            counts["exact_pairs"] += exact_count
            counts["near_pairs"] += matched_count - exact_count
            counts["cross_root_matching_pairs"] += int((candidates & cross).sum())
            if len(matches) >= max_pairs:
                continue
            # Per-row iteration bounds temporary indices even for identical datasets.
            for a in range(len(left)):
                for b in np.flatnonzero(candidates[a]):
                    lrec, rrec = records[i + a], records[j + int(b)]
                    is_exact = lrec.sha256 == rrec.sha256
                    is_cross = lrec.root != rrec.root
                    if len(matches) < max_pairs:
                        matches.append(
                            Match(
                                lrec.id,
                                rrec.id,
                                float(scores[a, b]),
                                "exact" if is_exact else "near",
                                is_cross,
                            )
                        )
                    if len(matches) >= max_pairs:
                        break
                if len(matches) >= max_pairs:
                    break
    return matches, counts


def exact_groups(records: list[ImageRecord], scope: str) -> list[list[int]]:
    """Only byte-identical images are grouped; similarity is not transitive."""
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sha256].append(record)
    return [
        [r.id for r in group]
        for group in grouped.values()
        if len(group) > 1 and (scope == "all" or len({r.root for r in group}) > 1)
    ]
