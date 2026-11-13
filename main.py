"""CLI entry point for batch ML drift and data quality monitoring."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config import DriftTrackerConfig
from src.pipeline import DriftTracker
from src.utils import generate_demo_datasets, load_csv_pair

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_REFERENCE = DEFAULT_DATA_DIR / "reference.csv"
DEFAULT_CURRENT = DEFAULT_DATA_DIR / "production_drifted.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch ML observability tool tracking feature, prediction, and data-quality drift."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to reference CSV file (defaults to bundled data/reference.csv).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="Path to current production CSV file (defaults to bundled data/production_drifted.csv).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run using freshly generated synthetic demo datasets (or bundled demo CSVs).",
    )
    parser.add_argument(
        "--prediction-col",
        default="prediction",
        help="Column name for discrete/continuous model predictions (if present).",
    )
    parser.add_argument(
        "--proba-col",
        default="pred_proba",
        help="Column name for predicted probabilities/scores (if present).",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Optional ground-truth label column excluded from feature drift checks.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full structured DriftReport as JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to write the Markdown summary report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when generating synthetic demo data.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo:
        ref_df, cur_df = generate_demo_datasets(seed=args.seed)
    elif args.reference is not None or args.current is not None:
        if args.reference is None or args.current is None:
            parser.error("--reference and --current must both be provided together")
        ref_df, cur_df = load_csv_pair(args.reference, args.current)
    elif DEFAULT_REFERENCE.exists() and DEFAULT_CURRENT.exists():
        ref_df, cur_df = load_csv_pair(DEFAULT_REFERENCE, DEFAULT_CURRENT)
    else:
        ref_df, cur_df = generate_demo_datasets(seed=args.seed)

    tracker = DriftTracker(
        config=DriftTrackerConfig(),
        prediction_col=args.prediction_col,
        proba_col=args.proba_col,
        target_col=args.target_col,
    )
    report = tracker.compare(ref_df, cur_df)

    markdown = report.to_markdown()
    print(markdown)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"\nSaved JSON report to {args.output_json}")

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
        print(f"Saved Markdown report to {args.output_markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
