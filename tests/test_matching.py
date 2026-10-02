import math

import numpy as np
import pytest

from image_similarity_detector.matching import compare, exact_groups
from image_similarity_detector.models import ImageRecord


def records(n):
    return [
        ImageRecord(
            i, f"/{i}.png", "train" if i % 2 else "test", f"{i}.png", f"sha-{i}", 10, 10, 100
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("metric", ["cosine", "hamming"])
@pytest.mark.parametrize("scope", ["all", "cross-root"])
@pytest.mark.parametrize("block_size", [1, 2, 7, 30])
def test_blocked_matches_agree_with_independent_pairwise_reference(metric, scope, block_size):
    rng = np.random.default_rng(19)
    features = (
        rng.integers(0, 2, (17, 32)).astype(np.float32)
        if metric == "hamming"
        else rng.normal(size=(17, 32)).astype(np.float32)
    )
    items = records(len(features))
    threshold = 0.49 if metric == "hamming" else 0.15
    expected = {}
    compared = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if scope == "cross-root" and items[i].root == items[j].root:
                continue
            compared += 1
            a, b = features[i].astype(float), features[j].astype(float)
            score = (
                1 - sum(x != y for x, y in zip(a, b, strict=True)) / len(a)
                if metric == "hamming"
                else sum(a * b) / math.sqrt(sum(a * a) * sum(b * b))
            )
            if score >= threshold:
                expected[i, j] = score
    found, totals = compare(
        features,
        items,
        metric=metric,
        threshold=threshold,
        scope=scope,
        block_size=block_size,
        max_pairs=1000,
    )
    assert {(m.left, m.right) for m in found} == set(expected)
    assert totals["pairs_compared"] == compared
    assert totals["matching_pairs"] == len(expected)
    for match in found:
        assert match.similarity == pytest.approx(expected[match.left, match.right], abs=1e-6)


def test_similar_chain_is_not_a_duplicate_group():
    angles = np.radians([0, 20, 40])
    features = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    found, _ = compare(
        features,
        records(3),
        metric="cosine",
        threshold=0.93,
        scope="all",
        block_size=2,
        max_pairs=10,
    )
    assert {(m.left, m.right) for m in found} == {(0, 1), (1, 2)}
    assert exact_groups(records(3), "all") == []


def test_storage_limit_does_not_truncate_counts():
    features = np.ones((20, 16), dtype=np.float32)
    found, counts = compare(
        features, records(20), metric="hamming", threshold=1, scope="all", block_size=3, max_pairs=2
    )
    assert len(found) == 2
    assert counts["matching_pairs"] == 190
    assert counts["pairs_compared"] == 190
    assert counts["near_pairs"] == 190
    assert counts["cross_root_matching_pairs"] == 100


def test_exact_bytes_are_independent_of_cosine_rounding():
    items = [ImageRecord(i, f"/{i}", "train", str(i), "same-digest", 10, 10, 10) for i in range(3)]
    found, counts = compare(
        np.array([[1, 2], [1, 2], [1, 2]], dtype=np.float32),
        items,
        metric="cosine",
        threshold=1,
        scope="all",
        block_size=1,
        max_pairs=10,
    )
    assert len(found) == 3
    assert all(m.kind == "exact" and m.similarity == 1 for m in found)
    assert counts["exact_pairs"] == 3
    assert exact_groups(items, "all") == [[0, 1, 2]]


def test_zero_cosine_vector_rejected():
    with pytest.raises(ValueError, match="zero"):
        compare(
            np.zeros((2, 3)),
            records(2),
            metric="cosine",
            threshold=0.9,
            scope="all",
            block_size=2,
            max_pairs=10,
        )
