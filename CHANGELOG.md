# Changelog

## Unreleased

- Report every input root with no readable images as an input issue, so empty or unsupported-only dataset splits cannot silently pass a strict cross-root gate. Preserve comparisons from readable roots.
- Distinguish zero compared pairs and incomplete no-match scans in HTML instead of suggesting threshold changes for missing input coverage.
- Add synthetic coverage, CLI exit-code, deterministic issue ordering, and source-immutability regressions. JSON remains schema 1.0 with the existing issue/status fields.

## 0.2.0 — 2026-10-01

- Replaced the editable script with an installable src-layout package, CLI, and library API.
- Added default offline dHash and optional batched ResNet50 with ImageNet V2 preprocessing.
- Added named inputs, cross-root checks, validated settings, and dataset gate exit codes.
- Added content-addressed cache with corruption recovery.
- Replaced full similarity allocation with block comparison and complete aggregate counts.
- Added capped pair retention, self-contained HTML review, and JSON schema 1.0.
- Separated exact identity from near similarity; only exact files are grouped.
- Removed transitive similarity deletion estimates, random sampling, and collage-only review.
- Added regression tests, multi-platform CI definition, wheel smoke tests, and synthetic demo.
- Rewrote English/Traditional Chinese documentation with usage, architecture, and limitations.

Neural weights/preprocessing changed; recalibrate thresholds. See [migration](docs/USAGE.md#migration-from-the-original-script).

## Original script — 2025-02

ResNet50 features, a full similarity matrix, cross-folder checks, JSON, and collage generation.

