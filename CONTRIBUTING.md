# Contributing

Use Python 3.10+ and focused changes tied to observable behavior.

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
ruff format --check src tests examples
python -m pytest
python -m build
```

Install `".[neural]"` for the optional neural architecture test. It uses random local weights to avoid downloads and does not measure pretrained detection accuracy.

Include the command, OS/Python version, backend, and a minimal synthetic example in bug reports. Avoid attaching company datasets or reports with private paths.

Detection changes should use independent reference checks and cover scope boundaries, non-transitive similarity, exact identity, and retention limits. Report changes need desktop/mobile verification and filename escaping checks.

Update documentation and CHANGELOG.md. JSON semantic changes require a compatibility decision. Accuracy/performance claims require reproducible evidence.

Open a pull request describing the problem, behavior, and validation. Local Linux success does not establish Windows/macOS compatibility; those are checked in CI.

