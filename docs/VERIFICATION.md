# Verification record

Executed on 2026-10-01 in Linux with Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0.

## Completed checks

- 44 pytest cases passed, including block results checked against an independent pairwise reference, non-transitive similarity, retention limits, exact identity, cross-root scope, invalid settings, source immutability, cache invalidation, corrupt images, EXIF orientation, CLI exits, HTML escaping and report limits.
- Ruff lint and formatting checks passed.
- Source distribution and wheel built with `python -m build --no-isolation`.
- Wheel installed into a separate environment and executed from `/tmp`, outside the source checkout. The installed CLI generated both JSON and HTML from the synthetic fixture; report assets were available from the installed package.
- Default demo: 8 images, 28 eligible pairs, 6 matching pairs, including 2 exact and 4 near candidates.
- Actual ImageNet V2 ResNet50 weights downloaded and evaluated on CPU with torch 2.14.1+cpu / torchvision 0.29.1+cpu. Two-image batch features matched individual inference within `rtol=1e-4`, `atol=1e-5`. The 8-image demo at threshold 0.99 produced 2 exact and 2 near pairs.
- Chromium checked the generated HTML at 1440, 1000, 720 and 390 px widths. Exact/near/cross-root filters, path search and empty-filter feedback passed. No horizontal overflow, remote resource requests or browser errors occurred.

## Reproduce the browser checks

The included sample HTML contains synthetic previews only. From the repository root:

```bash
python -m pip install playwright
python -m playwright install chromium --only-shell
python examples/verify_report.py
```

This checks the fixed demo and regenerates its desktop/mobile screenshots. Playwright is not a runtime dependency of the scanner.

## Limits of this verification

The demo is a functional fixture, not a measured detection benchmark. No claim of accuracy on company CCTV or other real datasets follows from these counts. CUDA/MPS and cross-platform execution were not locally tested. The GitHub workflow provides separate Windows/macOS/Linux and Python-version checks; inspect its results for the submitted commit.

The neural unit test uses local random weights to avoid CI model downloads. The separate pretrained check above exercised real weights, but still does not establish detection accuracy.
