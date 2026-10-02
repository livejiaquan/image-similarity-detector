from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from image_similarity_detector.cache import FeatureCache
from image_similarity_detector.features import DHashEncoder, load_image
from image_similarity_detector.models import InputRoot, ScanConfig
from image_similarity_detector.pipeline import scan


def config(dataset, **kwargs):
    train, test = dataset
    return ScanConfig((InputRoot("train", train), InputRoot("test", test)), **kwargs)


def test_cross_root_scan_and_source_immutability(dataset):
    originals = {p: p.read_bytes() for root in dataset for p in root.iterdir()}
    result = scan(config(dataset, scope="cross-root"))
    assert result.summary["images_scanned"] == 4
    assert result.summary["pairs_compared"] == 4
    assert result.summary["exact_pairs"] == 1
    assert len(result.exact_groups) == 1
    assert all(match.cross_root for match in result.matches)
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_bad_image_is_reported_without_losing_valid_matches(dataset):
    (dataset[0] / "broken.jpg").write_bytes(b"not an image")
    result = scan(config(dataset))
    assert result.summary["status"] == "partial"
    assert result.summary["images_discovered"] == 5
    assert result.summary["images_scanned"] == 4
    assert len(result.issues) == 1
    assert result.summary["exact_pairs"] == 1


def test_cache_reuse_and_invalid_cache_recovery(dataset, tmp_path):
    cached_config = config(dataset, cache_dir=tmp_path / "cache", batch_size=1)
    first = scan(cached_config)
    second = scan(cached_config)
    assert second.summary["cache_hits"] == 4
    assert first.matches == second.matches
    for cache_file in (tmp_path / "cache").rglob("*.npy"):
        cache_file.write_bytes(b"broken cache")
    repaired = scan(cached_config)
    assert repaired.summary["cache_hits"] < 4
    assert repaired.matches == first.matches


def test_cache_invalidation_when_file_content_changes(dataset, tmp_path):
    settings = config(dataset, cache_dir=tmp_path / "cache")
    scan(settings)
    Image.new("RGB", (100, 80), "black").save(dataset[1] / "copy.png")
    result = scan(settings)
    assert result.summary["cache_hits"] == 3
    assert result.summary["exact_pairs"] == 0


def test_batches_encode_multiple_images(dataset):
    class RecordingEncoder(DHashEncoder):
        def __init__(self):
            self.batch_lengths = []

        def encode(self, images):
            self.batch_lengths.append(len(images))
            return super().encode(images)

    encoder = RecordingEncoder()
    scan(config(dataset, batch_size=3), encoder=encoder)
    assert encoder.batch_lengths == [3, 1]


@pytest.mark.parametrize(
    "setting,value",
    [
        ("threshold", float("nan")),
        ("threshold", 1.1),
        ("block_size", 0),
        ("batch_size", -1),
        ("max_pairs", 0),
        ("report_pairs", 0),
    ],
)
def test_bad_settings_fail_before_scan(dataset, setting, value):
    with pytest.raises(ValueError):
        scan(replace(config(dataset), **{setting: value}))


def test_overlapping_roots_rejected(dataset):
    root = dataset[0]
    with pytest.raises(ValueError, match="overlap"):
        scan(ScanConfig((InputRoot("a", root), InputRoot("b", root))))


def test_empty_dataset_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="No supported"):
        scan(ScanConfig((InputRoot("empty", tmp_path),)))


def test_symlink_cannot_escape_input_root(dataset, tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("RGB", (20, 20), "red").save(outside)
    try:
        (dataset[0] / "link.png").symlink_to(outside)
    except OSError:
        pytest.skip("Platform does not permit symbolic link creation.")
    result = scan(config(dataset))
    assert result.summary["images_scanned"] == 4
    assert any("Symbolic link" in issue.reason for issue in result.issues)


def test_cache_rejects_pickle_nonfinite_and_wrong_dimensions(tmp_path):
    cache = FeatureCache(tmp_path, "backend-a", 3)
    path = cache.directory / "a.npy"
    for value in (
        np.array([np.nan, 0, 1]),
        np.array([1, 2]),
        np.array(["a", "b", "c"]),
        np.array([object(), object(), object()], dtype=object),
    ):
        np.save(path, value)
        assert cache.get("a") is None
    other = FeatureCache(tmp_path, "backend-b", 3)
    assert cache.directory != other.directory


def test_exif_orientation_is_normalized(tmp_path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (40, 20), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    with load_image(str(path)) as loaded:
        assert loaded.size == (20, 40)
