"""Self-contained offline review reports. Dataset names are always HTML-escaped."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from string import Template

from .discovery import file_digest
from .features import load_image
from .models import ImageRecord, ScanResult


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def thumbnail(record: ImageRecord) -> str:
    try:
        if file_digest(Path(record.path)) != record.sha256:
            return '<div class="preview-unavailable">Source changed since scan</div>'
        with load_image(record.path) as image:
            image.thumbnail((480, 300))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=78)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f'<img src="data:image/jpeg;base64,{encoded}" alt="{escape(record.relative_path)}">'
    except (OSError, ValueError, Warning):
        return '<div class="preview-unavailable">Preview unavailable</div>'


def render_html(result: ScanResult, report_pairs: int = 100) -> str:
    records = {record.id: record for record in result.images}
    previews: dict[int, str] = {}
    cards = []
    for match in result.matches[:report_pairs]:
        left, right = records[match.left], records[match.right]
        figures = []
        for record in (left, right):
            if record.id not in previews:
                previews[record.id] = thumbnail(record)
            figures.append(
                f"<figure>{previews[record.id]}<figcaption><span>{escape(record.root)}</span>"
                f"<strong>{escape(record.relative_path)}</strong>"
                f"<small>{record.width} × {record.height} · {record.size_bytes:,} bytes</small>"
                "</figcaption></figure>"
            )
        cross_badge = '<span class="badge cross">Cross-root</span>' if match.cross_root else ""
        description = "SHA-256 identical" if match.kind == "exact" else "Visual review required"
        cards.append(
            f'<article class="match-card" data-kind="{match.kind}" '
            f'data-cross="{str(match.cross_root).lower()}">'
            f'<div class="card-head"><div><span class="badge {match.kind}">'
            f"{'Exact duplicate' if match.kind == 'exact' else 'Near-match candidate'}</span>"
            f'{cross_badge}</div><span class="score">{match.similarity * 100:.2f}'
            f"<small> / 100 similarity</small></span></div>"
            f'<div class="image-pair">{"".join(figures)}</div>'
            f'<div class="card-foot">{description}</div></article>'
        )
    if not cards:
        if result.summary["pairs_compared"] == 0:
            title = "No image pairs were compared."
            guidance = "Check input issues and provide readable images in the selected scope. "
        elif result.issues:
            title = "No matches among readable inputs."
            guidance = "This scan is incomplete. Resolve input issues and scan again. "
        else:
            title = "No matches at this threshold."
            guidance = "Try another backend or threshold if you expected similar images. "
        cards.append(
            f'<div class="empty-state"><strong>{title}</strong><p>{guidance}'
            "This result does not prove that the dataset is free of contamination.</p></div>"
        )
    groups = []
    for group in result.exact_groups:
        names = ", ".join(f"{records[i].root}/{records[i].relative_path}" for i in group)
        groups.append(f"<li>{escape(names)}</li>")
    issues = (
        "".join(
            f"<li><strong>{escape(Path(issue.path).name)}</strong> — {escape(issue.reason)}</li>"
            for issue in result.issues[:100]
        )
        or "<li>No unreadable or skipped inputs reported.</li>"
    )
    assets = files("image_similarity_detector").joinpath("assets")
    script = assets.joinpath("report.js").read_text(encoding="utf-8")
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    template = Template(assets.joinpath("report.html").read_text(encoding="utf-8"))
    summary = result.summary
    return template.substitute(
        css=assets.joinpath("report.css").read_text(encoding="utf-8"),
        js=script,
        script_hash=script_hash,
        version=escape(result.tool_version),
        date=escape(result.started_at[:19].replace("T", " ") + " UTC"),
        status=escape(summary["status"].upper()),
        status_class="partial" if summary["status"] == "partial" else "complete",
        images=f"{summary['images_scanned']:,}",
        exact=f"{summary['exact_pairs']:,}",
        near=f"{summary['near_pairs']:,}",
        cross=f"{summary['cross_root_matching_pairs']:,}",
        issue_count=summary["issues_count"],
        backend=escape(result.backend["name"]),
        metric=escape(result.backend["metric"]),
        threshold=escape(result.config["threshold"]),
        duration=escape(result.duration_seconds),
        scope=escape(result.config["scope"]),
        compared=f"{summary['pairs_compared']:,}",
        cache=summary["cache_hits"],
        cards="".join(cards),
        shown=min(len(result.matches), report_pairs),
        total=f"{summary['matching_pairs']:,}",
        stored=len(result.matches),
        truncated="Yes" if summary["pairs_truncated"] else "No",
        warnings="".join(f"<li>{escape(w)}</li>" for w in result.warnings),
        groups="".join(groups) or "<li>No byte-identical groups in the selected scope.</li>",
        issues=issues,
        issue_note="Showing the first 100 issues; see JSON for all."
        if len(result.issues) > 100
        else "",
        inputs="".join(f"<li>{escape(root['label'])}</li>" for root in result.config["inputs"]),
    )


def write_reports(result: ScanResult, output: Path, report_pairs: int = 100) -> Path:
    output = output.resolve()
    for root in result.config["inputs"]:
        source = Path(root["path"]).resolve()
        if output == source or source in output.parents:
            raise ValueError("Reports must be written outside all input roots.")
    output.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("scan-%Y%m%d-%H%M%S-")
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=output))
    try:
        (directory / "report.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (directory / "report.html").write_text(render_html(result, report_pairs), encoding="utf-8")
    except Exception:
        shutil.rmtree(directory)
        raise
    return directory
