"""Public command-line interface and stable exit codes."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .models import InputRoot, ScanConfig
from .pipeline import scan
from .reporting import write_reports


def input_root(value: str) -> InputRoot:
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path = value
        label = Path(path).name
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "Use --input LABEL=PATH, for example --input train=./train."
        )
    return InputRoot(label, Path(path).expanduser())


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="image-similarity",
        description="Audit local image datasets. Source files are never modified.",
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("scan", help="Scan image folders and write JSON + HTML reports.")
    command.add_argument(
        "--input",
        action="append",
        required=True,
        type=input_root,
        help="Input directory, optionally LABEL=PATH. Repeat for multiple roots.",
    )
    command.add_argument(
        "--output", type=Path, default=Path("results"), help="Report parent directory."
    )
    command.add_argument("--backend", choices=["dhash", "resnet50"], default="dhash")
    command.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Similarity threshold, 0–1. Backend-specific; default 0.95.",
    )
    command.add_argument("--scope", choices=["all", "cross-root"], default="all")
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--block-size", type=int, default=256)
    command.add_argument(
        "--max-pairs", type=int, default=10_000, help="Maximum stored match records."
    )
    command.add_argument("--report-pairs", type=int, default=100, help="Maximum HTML review cards.")
    command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    command.add_argument("--cache-dir", type=Path, help="Opt-in content-addressed feature cache.")
    command.add_argument(
        "--fail-on-matches",
        action="store_true",
        help="Exit 1 when matches are found, for dataset CI checks.",
    )
    command.add_argument(
        "--strict", action="store_true", help="Exit 2 for unreadable/skipped input or empty roots."
    )
    command.add_argument("--verbose", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    config = ScanConfig(
        roots=tuple(args.input),
        backend=args.backend,
        threshold=args.threshold,
        scope=args.scope,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_pairs=args.max_pairs,
        report_pairs=args.report_pairs,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    try:
        # Validate destinations before expensive encoding, and keep sources read-only.
        config.validate()
        destinations = [args.output] + ([args.cache_dir] if args.cache_dir else [])
        for destination in destinations:
            resolved = destination.expanduser().resolve()
            for input_dir in config.roots:
                source = input_dir.path.resolve()
                if resolved == source or source in resolved.parents:
                    raise ValueError(
                        "Output and cache directories must be outside all input roots."
                    )
        result = scan(config)
        report_dir = write_reports(result, args.output.expanduser(), args.report_pairs)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Scan interrupted.", file=sys.stderr)
        return 130
    summary = result.summary
    print(
        f"Images: {summary['images_scanned']} | Matches: {summary['matching_pairs']} "
        f"| Exact: {summary['exact_pairs']} | Near: {summary['near_pairs']}"
    )
    print(
        f"Cross-root matches: {summary['cross_root_matching_pairs']} "
        f"| Issues: {summary['issues_count']} | Cache hits: {summary['cache_hits']}"
    )
    if summary["pairs_truncated"]:
        print(f"Match storage limited to {summary['pairs_stored']}; see report warnings.")
    print(f"Review: {report_dir / 'report.html'}")
    print(f"JSON:   {report_dir / 'report.json'}")
    if args.strict and result.issues:
        return 2
    return 1 if args.fail_on_matches and summary["matching_pairs"] else 0
