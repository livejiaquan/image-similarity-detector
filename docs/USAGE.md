# CLI reference

Install with `python -m pip install .`. Use `image-similarity scan` or `python -m image_similarity_detector scan`. `--version` is a top-level flag; `scan --help` lists scan options.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input LABEL=PATH` | Required, repeatable | Named input; bare paths use the directory name |
| `--output PATH` | `results` | Parent for unique report directories |
| `--backend` | `dhash` | `dhash` or `resnet50` |
| `--threshold` | `0.95` | Inclusive threshold from 0 to 1 |
| `--scope` | `all` | `all` or `cross-root` |
| `--batch-size` | `32` | Maximum encoded images per batch |
| `--block-size` | `256` | Maximum rows per side of a comparison block |
| `--max-pairs` | `10000` | Retained matches; aggregate counts stay complete |
| `--report-pairs` | `100` | Maximum HTML cards from retained records |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps`; dHash uses CPU |
| `--cache-dir PATH` | Disabled | Opt-in feature cache |
| `--fail-on-matches` | Disabled | Exit 1 when matches are found |
| `--strict` | Disabled | Exit 2 for skipped/unreadable input |
| `--verbose` | Disabled | Encoding and comparison progress |

Labels must be unique; paths with spaces should be quoted: `--input "train=./my data/train"`. Roots must exist, must not overlap, and cannot be symlinks. Output/cache inside roots is rejected. Cross-root mode requires at least two roots.

## Dataset gate

```bash
image-similarity scan --input train=./data/train --input test=./data/test \
  --scope cross-root --strict --fail-on-matches --output ./audit-results
```

| Exit | Meaning |
| --- | --- |
| 0 | Finished; matches are informational unless the gate was requested |
| 1 | Finished with matches under `--fail-on-matches` |
| 2 | Invalid settings, no readable images, execution/report error, or strict input issues |
| 130 | User interruption |

A partial scan still writes a report when readable images exist. Strict issues take precedence over the match gate. This checks the selected metric/threshold; it does not certify dataset independence.

## Tuning

A dHash threshold of 0.95 accepts at most 12 different bits out of 256. Solid-color images may share hashes despite different colors. Crops and rotations can cause missed matches. ResNet cosine similarity can reflect shared subjects rather than duplicates. Review positive and negative examples before adopting a threshold.

Reduce batch size for large decoded images and block size for comparison memory. The pair cap bounds serialization, not quadratic comparison work.

## Library API

```python
from pathlib import Path
from image_similarity_detector.models import InputRoot, ScanConfig
from image_similarity_detector.pipeline import scan
from image_similarity_detector.reporting import write_reports

result = scan(ScanConfig(
    roots=(InputRoot("images", Path("./dataset/images")),),
    cache_dir=Path("./feature-cache"),
))
directory = write_reports(result, Path("./results"))
print(result.summary)
```

The beta API is separate from the JSON schema. An encoder can be reused through `scan(config, encoder=...)`; its output must be finite and aligned with inputs.

## Migration from the original script

- Replace `python image_similarity_detector.py` with the installed CLI.
- Replace edits to `folder_list` with repeated `--input` arguments.
- Replace `inter_folder` with `--scope cross-root`.
- Select `--backend resnet50` for neural features. V2 weights/preprocessing differ from the original stretched 224 × 224 transform; recalibrate thresholds.
- Exhaustive scanning and browsable cards replace random sampling and collage-only review.
- Schema 1.0 replaces the old flat report; `photos_to_remove` is removed because similarity is not a safe deletion rule.

