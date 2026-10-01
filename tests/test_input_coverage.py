"""A missing dataset split must not silently pass a strict audit gate."""

import json
from dataclasses import replace

import pytest
from PIL import Image

from image_similarity_detector.cli import main
from image_similarity_detector.models import InputRoot, ScanConfig
from image_similarity_detector.pipeline import scan
from image_similarity_detector.reporting import render_html


@pytest.mark.parametrize("scope", ["all", "cross-root"])
@pytest.mark.parametrize("contents", ["empty", "unsupported", "corrupt"])
def test_root_without_readable_images_is_reported(dataset, tmp_path, scope, contents):
    missing = tmp_path / "missing split"
    missing.mkdir()
    if contents == "unsupported":
        (missing / "notes.txt").write_text("No supported images here.", encoding="utf-8")
    elif contents == "corrupt":
        (missing / "broken.png").write_bytes(b"not an image")
    roots = (
        InputRoot("train", dataset[0]),
        InputRoot("test", dataset[1]),
        InputRoot("validation", missing),
    )
    originals = {p: p.read_bytes() for root in roots for p in root.path.iterdir()}
    result = scan(ScanConfig(roots, scope=scope))
    complete = scan(ScanConfig(roots[:2], scope=scope))

    assert result.summary["status"] == "partial"
    assert result.summary["images_scanned"] == 4
    assert result.summary["images_discovered"] == (5 if contents == "corrupt" else 4)
    assert result.summary["issues_count"] == (2 if contents == "corrupt" else 1)
    root_issues = [issue for issue in result.issues if issue.path == str(missing.resolve())]
    assert len(root_issues) == 1
    assert "validation" in root_issues[0].reason
    assert "No readable images" in root_issues[0].reason
    assert result.matches == complete.matches
    assert result.exact_groups == complete.exact_groups
    assert result.summary["pairs_compared"] == complete.summary["pairs_compared"]
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_empty_root_issues_are_deterministic_and_identify_named_inputs(dataset, tmp_path):
    first, second = tmp_path / "one" / "images", tmp_path / "two" / "images"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    config = ScanConfig(
        (
            InputRoot("train", dataset[0]),
            InputRoot("validation", first),
            InputRoot("test", second),
        )
    )
    forward = scan(config)
    reverse = scan(replace(config, roots=tuple(reversed(config.roots))))
    assert forward.issues == reverse.issues
    assert {issue.path for issue in forward.issues} == {str(first), str(second)}
    document = render_html(forward)
    assert "input &#x27;validation&#x27;" in document
    assert "input &#x27;test&#x27;" in document


def test_nested_images_count_toward_input_root_coverage(dataset, tmp_path):
    nested = tmp_path / "validation" / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (10, 10), "red").save(nested / "image.png")
    result = scan(
        ScanConfig((InputRoot("train", dataset[0]), InputRoot("validation", nested.parent)))
    )
    assert result.summary["status"] == "complete"
    assert result.issues == []
    assert any(record.relative_path == "nested/image.png" for record in result.images)


@pytest.mark.parametrize("strict,expected_exit", [(False, 0), (True, 2)])
def test_cross_root_gate_and_reports_surface_empty_split(tmp_path, strict, expected_exit):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    Image.new("RGB", (10, 10), "red").save(train / "one.png")
    output = tmp_path / "reports"
    args = [
        "scan",
        "--input",
        f"train={train}",
        "--input",
        f"test={test}",
        "--scope",
        "cross-root",
        "--fail-on-matches",
        "--output",
        str(output),
    ]
    if strict:
        args.append("--strict")
    assert main(args) == expected_exit
    report = next(output.glob("*/report.json"))
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["summary"]["status"] == "partial"
    assert data["summary"]["pairs_compared"] == 0
    assert data["summary"]["matching_pairs"] == 0
    assert data["summary"]["issues_count"] == 1
    document = report.with_suffix(".html").read_text(encoding="utf-8")
    assert "No image pairs were compared." in document
    assert "PARTIAL" in document
    assert "No matches at this threshold." not in document


def test_empty_split_strict_issue_takes_precedence_over_matches(dataset, tmp_path):
    missing = tmp_path / "validation"
    missing.mkdir()
    assert (
        main(
            [
                "scan",
                "--input",
                f"train={dataset[0]}",
                "--input",
                f"test={dataset[1]}",
                "--input",
                f"validation={missing}",
                "--scope",
                "cross-root",
                "--strict",
                "--fail-on-matches",
                "--output",
                str(tmp_path / "reports"),
            ]
        )
        == 2
    )


def test_all_empty_roots_still_fail_without_a_report(tmp_path, capsys):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    output = tmp_path / "reports"
    assert (
        main(
            [
                "scan",
                "--input",
                f"train={train}",
                "--input",
                f"test={test}",
                "--scope",
                "cross-root",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    assert "No supported images" in capsys.readouterr().err


@pytest.mark.parametrize("missing_split", [False, True])
def test_no_match_report_distinguishes_incomplete_scan(dataset, tmp_path, missing_split):
    roots = [InputRoot("train", dataset[0])]
    if missing_split:
        missing = tmp_path / "validation"
        missing.mkdir()
        roots.append(InputRoot("validation", missing))
    result = scan(ScanConfig(tuple(roots), threshold=1))
    assert result.summary["pairs_compared"] == 1
    assert result.matches == []
    document = render_html(result)
    if missing_split:
        assert "No matches among readable inputs." in document
        assert "Resolve input issues and scan again." in document
        assert "Try another backend or threshold" not in document
    else:
        assert "No matches at this threshold." in document
        assert "Try another backend or threshold" in document


def test_relative_unicode_empty_root_gets_absolute_issue_path(dataset, tmp_path, monkeypatch):
    missing = tmp_path / "驗證 split"
    missing.mkdir()
    monkeypatch.chdir(tmp_path)
    result = scan(
        ScanConfig(
            (
                InputRoot("train", dataset[0]),
                InputRoot("驗證", missing.relative_to(tmp_path)),
            )
        )
    )
    assert len(result.issues) == 1
    assert result.issues[0].path == str(missing.resolve())
    assert "驗證" in result.issues[0].reason
