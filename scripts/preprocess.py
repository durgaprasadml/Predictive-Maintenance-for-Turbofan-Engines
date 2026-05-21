"""
main.py — Entry point for the C-MAPSS FD001 preprocessing pipeline.

Usage
─────
    python main.py                        # defaults (data/ directory)
    python main.py --data-dir ./cmapss    # custom data path
    python main.py --window 50            # custom sequence length
    python main.py --no-clip              # disable RUL clipping

The script runs the full pipeline and prints a diagnostic summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from app.core.config import DatasetConfig
from scripts.data_pipeline import run_pipeline


def _configure_logging() -> None:
    """Set up console logging with a clean format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NASA C-MAPSS FD001 for RUL prediction.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train_FD001.txt, test_FD001.txt, RUL_FD001.txt",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Sliding window length for sequence generation (default: 30)",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=125,
        help="Piecewise-linear RUL cap (default: 125)",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable RUL clipping entirely",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of engines held out for validation (default: 0.2)",
    )
    return parser.parse_args()


def _print_summary(result) -> None:
    """Print a concise diagnostic summary of pipeline outputs."""
    divider = "─" * 60

    print(f"\n{divider}")
    print("  📊  PIPELINE SUMMARY")
    print(divider)

    print(f"\n  Train DataFrame     : {result.train_df.shape}")
    print(f"  Test DataFrame      : {result.test_df.shape}")
    print(f"  Features retained   : {len(result.feature_cols)}")
    print(f"  Sensors dropped     : {result.dropped_sensors or 'none'}")

    print(f"\n  {'Array':<20} {'Shape':<30} {'dtype'}")
    print(f"  {'─'*20} {'─'*30} {'─'*10}")
    for name, arr in [
        ("X_train", result.X_train),
        ("y_train", result.y_train),
        ("X_val", result.X_val),
        ("y_val", result.y_val),
        ("X_test", result.X_test),
        ("y_test", result.y_test),
    ]:
        print(f"  {name:<20} {str(arr.shape):<30} {arr.dtype}")

    print(f"\n  Train RUL range     : [{result.y_train.min():.0f}, {result.y_train.max():.0f}]")
    print(f"  Val   RUL range     : [{result.y_val.min():.0f}, {result.y_val.max():.0f}]")
    print(f"  Test  RUL range     : [{result.y_test.min():.0f}, {result.y_test.max():.0f}]")

    print(f"\n  Scaler feature range: {result.scaler.feature_range}")
    print(f"  Scaler fitted on    : {result.scaler.n_features_in_} features")

    print(f"\n{divider}")
    print("  ✅  Ready for model training")
    print(f"{divider}\n")


def main() -> None:
    _configure_logging()
    args = _parse_args()

    cfg = DatasetConfig(
        data_dir=args.data_dir,
        sequence_length=args.window,
        rul_clip_value=None if args.no_clip else args.clip,
        test_size=args.test_size,
    )

    result = run_pipeline(cfg)
    _print_summary(result)


if __name__ == "__main__":
    main()
