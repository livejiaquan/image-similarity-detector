# JSON schema 1.0

UTF-8 JSON. Schema and package versions are independent; non-finite numbers are not serialized.

| Field | Meaning |
| --- | --- |
| schema_version | `"1.0"` |
| tool_version | Package version |
| started_at | UTC ISO 8601 |
| duration_seconds | Scan time, excluding report rendering |
| config | Inputs and requested options |
| backend | Metric, dimensions, encoder fingerprint, resolved device, weights if applicable |
| images | All successfully decoded records |
| matches | Retained pairs, bounded by max_pairs |
| exact_groups | Image IDs sharing a SHA-256 digest in the selected scope |
| issues | Skipped input paths and reasons, including roots with no readable images |
| summary | Coverage, complete counts, status, cache hits, truncation |
| warnings | Interpretation and limit notes |

An image has `id`, absolute `path`, named `root`, `relative_path`, `sha256`, EXIF-normalized `width`/`height`, and `size_bytes`. IDs are zero-based within one scan, not persistent asset identities.

A pair has `left`/`right` image IDs, `similarity`, `kind` (`exact` or `near`), and `cross_root`. Exact pairs score 1. Near scores use the selected backend; configured thresholds are [0, 1].

## Summary contract

- `status`: `complete` without input issues; `partial` otherwise. Complete does not guarantee every possible duplicate was detected.
- `images_discovered`: supported-extension files found, excluding symlinks.
- `images_scanned`: readable images included in comparisons.
- `issues_count`: input issues, including traversal, symlink, and empty/unreadable-root issues. A root with no readable images has its own directory-level issue in addition to any per-file errors; this is not a count of failed image files.
- `pairs_compared`: all eligible unordered pairs.
- `matching_pairs`: all qualifying visual or exact pairs.
- `exact_pairs` and `near_pairs`: partition matching_pairs.
- `cross_root_matching_pairs`: matches spanning named roots.
- `pairs_stored`: retained match count.
- `pairs_truncated`: true when some matches were counted but not retained.
- `exact_groups`: digest-group count independent of retention limits.
- `cache_hits`: vectors reused from cache, including reuse during the same run.

All mode compares N(N−1)/2 pairs; cross-root only compares distinct root labels. Decode failures reduce N. A named root with no readable images makes an otherwise usable scan partial. This uses the existing schema 1.0 issue/status fields; no fields or pair semantics change. A scan with no readable images anywhere still fails without a report. Consumers must inspect status/issues before interpreting zero matches, and check truncation before assuming the match list is exhaustive.

HTML filters operate only on displayed cards. JSON may retain more pairs than HTML, while totals may exceed both. There is no `photos_to_remove` field.

