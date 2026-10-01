import json
from dataclasses import replace
from pathlib import Path

from image_similarity_detector.cli import main
from image_similarity_detector.models import InputRoot, ScanConfig
from image_similarity_detector.pipeline import scan
from image_similarity_detector.reporting import render_html, write_reports


def args(dataset, output):
    return [
        "scan",
        "--input",
        f"train={dataset[0]}",
        "--input",
        f"test={dataset[1]}",
        "--output",
        str(output),
    ]


def test_cli_writes_valid_report_and_exit_codes(dataset, tmp_path, capsys):
    output = tmp_path / "reports"
    assert main(args(dataset, output)) == 0
    files = list(output.glob("*/report.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report["schema_version"] == "1.0"
    assert report["summary"]["exact_pairs"] == 1
    assert files[0].with_suffix(".html").exists()
    assert "Review:" in capsys.readouterr().out
    assert main(args(dataset, output) + ["--fail-on-matches"]) == 1
    assert len(list(output.glob("*/report.json"))) == 2


def test_strict_mode_still_emits_reviewable_partial_report(dataset, tmp_path):
    (dataset[0] / "broken.jpg").write_bytes(b"broken")
    output = tmp_path / "reports"
    assert main(args(dataset, output) + ["--strict"]) == 2
    report = json.loads(next(output.glob("*/report.json")).read_text())
    assert report["summary"]["status"] == "partial"


def test_output_and_cache_cannot_write_inside_sources(dataset, tmp_path, capsys):
    assert main(args(dataset, dataset[0] / "reports")) == 2
    assert not (dataset[0] / "reports").exists()
    assert (
        main(args(dataset, tmp_path / "reports") + ["--cache-dir", str(dataset[0] / "cache")]) == 2
    )
    assert not (dataset[0] / "cache").exists()
    assert "outside" in capsys.readouterr().err


def test_bad_paths_exit_two_without_traceback(tmp_path, capsys):
    assert main(["scan", "--input", "missing=" + str(tmp_path / "missing")]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_report_escapes_untrusted_filenames_and_embeds_no_remote_assets(dataset):
    malicious = dataset[0] / "image&report.png"
    malicious.write_bytes((dataset[0] / "original.png").read_bytes())
    result = scan(
        ScanConfig(
            (InputRoot("<script>alert(1)</script>", dataset[0]), InputRoot("test", dataset[1]))
        )
    )
    result.images = [
        replace(record, relative_path="<img src=x onerror=alert(1)>.png")
        if Path(record.path) == malicious
        else record
        for record in result.images
    ]
    document = render_html(result)
    assert "<img src=x onerror=alert(1)>" not in document
    assert "<script>alert(1)</script>" not in document
    assert "&lt;img src=x onerror=alert(1)&gt;" in document
    assert "Content-Security-Policy" in document
    assert "https://" not in document
    assert "data:image/jpeg;base64," in document


def test_report_limit_is_visible_and_json_retains_records(dataset, tmp_path):
    result = scan(
        ScanConfig(
            (InputRoot("train", dataset[0]), InputRoot("test", dataset[1])),
            threshold=0,
            max_pairs=3,
        )
    )
    directory = write_reports(result, tmp_path / "reports", report_pairs=1)
    document = (directory / "report.html").read_text()
    assert document.count('<article class="match-card"') == 1
    assert "1 cards · 3 stored pairs · 6 matches found" in document
    report = json.loads((directory / "report.json").read_text())
    assert len(report["matches"]) == 3
    assert report["summary"]["pairs_truncated"] is True


def test_changed_source_not_shown_as_old_preview(dataset):
    result = scan(ScanConfig((InputRoot("train", dataset[0]), InputRoot("test", dataset[1]))))
    Path(result.images[0].path).write_bytes(b"changed")
    assert "Source changed since scan" in render_html(result)
