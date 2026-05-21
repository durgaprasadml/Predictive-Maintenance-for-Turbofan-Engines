"""
train_models.py — Train and compare RF, XGBoost, and LSTM for RUL prediction.

Usage
─────
    python train_models.py                         # all defaults
    python train_models.py --epochs 80             # LSTM epochs
    python train_models.py --skip-lstm              # tree models only
    python train_models.py --output-dir results    # custom output dir

Outputs (saved to ./outputs/):
    models/
        random_forest.joblib
        xgboost.joblib
        lstm_rul.h5
        metadata.json          ← best model name + input shape
    plots/
        randomforest_predictions.png
        xgboost_predictions.png
        lstm_predictions.png
        lstm_training_history.png
        model_comparison.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from app.core.config import DatasetConfig
from scripts.data_pipeline import run_pipeline, PipelineResult
from scripts.evaluate import (
    Metrics,
    compare_models,
    compute_metrics,
    plot_comparison,
    plot_predictions,
    plot_training_history,
)
from scripts.models import build_lstm, build_random_forest, build_xgboost, flatten_sequences

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Training routines
# ═══════════════════════════════════════════════════════════════════════

def train_random_forest(
    data: PipelineResult,
    output_dir: Path,
) -> Dict[str, Metrics]:
    """Train and evaluate a Random Forest regressor."""
    logger.info("=" * 60)
    logger.info("Training Random Forest")
    logger.info("=" * 60)

    X_train_flat = flatten_sequences(data.X_train)
    X_val_flat = flatten_sequences(data.X_val)
    X_test_flat = flatten_sequences(data.X_test)

    model = build_random_forest()

    t0 = time.time()
    model.fit(X_train_flat, data.y_train)
    elapsed = time.time() - t0
    logger.info("RF training time: %.1fs", elapsed)

    # Save model
    model_path = output_dir / "models" / "random_forest.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved → %s", model_path)

    # Evaluate on validation and test
    results = {}

    y_val_pred = model.predict(X_val_flat)
    val_metrics = compute_metrics(data.y_val, y_val_pred)
    results["val"] = val_metrics
    logger.info("RF Validation: %s", val_metrics)

    y_test_pred = model.predict(X_test_flat)
    test_metrics = compute_metrics(data.y_test, y_test_pred)
    results["test"] = test_metrics
    logger.info("RF Test: %s", test_metrics)

    # Plot
    plot_predictions(
        data.y_test, y_test_pred, "RandomForest",
        output_dir / "plots", test_metrics,
    )

    return {"RandomForest": test_metrics}


def train_xgboost(
    data: PipelineResult,
    output_dir: Path,
) -> Dict[str, Metrics]:
    """Train and evaluate an XGBoost regressor."""
    logger.info("=" * 60)
    logger.info("Training XGBoost")
    logger.info("=" * 60)

    X_train_flat = flatten_sequences(data.X_train)
    X_val_flat = flatten_sequences(data.X_val)
    X_test_flat = flatten_sequences(data.X_test)

    model = build_xgboost()

    t0 = time.time()
    model.fit(
        X_train_flat, data.y_train,
        eval_set=[(X_val_flat, data.y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    logger.info("XGB training time: %.1fs", elapsed)

    # Save model
    model_path = output_dir / "models" / "xgboost.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved → %s", model_path)

    # Evaluate
    results = {}

    y_val_pred = model.predict(X_val_flat)
    val_metrics = compute_metrics(data.y_val, y_val_pred)
    results["val"] = val_metrics
    logger.info("XGB Validation: %s", val_metrics)

    y_test_pred = model.predict(X_test_flat)
    test_metrics = compute_metrics(data.y_test, y_test_pred)
    results["test"] = test_metrics
    logger.info("XGB Test: %s", test_metrics)

    # Plot
    plot_predictions(
        data.y_test, y_test_pred, "XGBoost",
        output_dir / "plots", test_metrics,
    )

    return {"XGBoost": test_metrics}


def train_lstm(
    data: PipelineResult,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
) -> Dict[str, Metrics]:
    """Train and evaluate an LSTM model."""
    import tensorflow as tf

    logger.info("=" * 60)
    logger.info("Training LSTM (epochs=%d, batch=%d)", epochs, batch_size)
    logger.info("=" * 60)

    input_shape = (data.X_train.shape[1], data.X_train.shape[2])
    model = build_lstm(input_shape=input_shape)
    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info("LSTM training time: %.1fs", elapsed)

    # Save model
    model_path = output_dir / "models" / "lstm_rul.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info("Saved → %s", model_path)

    # Evaluate
    results = {}

    y_val_pred = model.predict(data.X_val, verbose=0).flatten()
    val_metrics = compute_metrics(data.y_val, y_val_pred)
    results["val"] = val_metrics
    logger.info("LSTM Validation: %s", val_metrics)

    y_test_pred = model.predict(data.X_test, verbose=0).flatten()
    test_metrics = compute_metrics(data.y_test, y_test_pred)
    results["test"] = test_metrics
    logger.info("LSTM Test: %s", test_metrics)

    # Plots
    plot_predictions(
        data.y_test, y_test_pred, "LSTM",
        output_dir / "plots", test_metrics,
    )
    plot_training_history(history.history, output_dir / "plots")

    return {"LSTM": test_metrics}


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def train_all(
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    skip_lstm: bool = False,
) -> None:
    """
    Run the full data pipeline, train all models, compare, and save results.
    """
    # ── 1. Data ─────────────────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("  PHASE 1 — Data Preprocessing")
    logger.info("━" * 60)
    data = run_pipeline()

    # ── 2. Train models ────────────────────────────────────────────
    all_metrics: Dict[str, Metrics] = {}

    logger.info("\n" + "━" * 60)
    logger.info("  PHASE 2 — Model Training")
    logger.info("━" * 60)

    rf_metrics = train_random_forest(data, output_dir)
    all_metrics.update(rf_metrics)

    xgb_metrics = train_xgboost(data, output_dir)
    all_metrics.update(xgb_metrics)

    if not skip_lstm:
        lstm_metrics = train_lstm(data, output_dir, epochs, batch_size)
        all_metrics.update(lstm_metrics)

    # ── 3. Compare ──────────────────────────────────────────────────
    logger.info("\n" + "━" * 60)
    logger.info("  PHASE 3 — Model Comparison")
    logger.info("━" * 60)

    table = compare_models(all_metrics)
    print(f"\n{table}\n")

    # Determine best model
    best_name = min(all_metrics, key=lambda k: all_metrics[k].rmse)
    best_m = all_metrics[best_name]
    print(f"  🏆 Best model: {best_name} (RMSE={best_m.rmse:.2f})\n")

    # Comparison plot
    plot_comparison(all_metrics, output_dir / "plots")

    # ── 4. Save metadata ───────────────────────────────────────────
    meta = {
        "best_model": best_name,
        "input_shape": [int(data.X_train.shape[1]), int(data.X_train.shape[2])],
        "feature_cols": data.feature_cols,
        "dropped_sensors": data.dropped_sensors,
        "results": {
            name: {"rmse": m.rmse, "mae": m.mae, "r2": m.r2}
            for name, m in all_metrics.items()
        },
    }
    meta_path = output_dir / "models" / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved → %s", meta_path)

    # ── Final summary ──────────────────────────────────────────────
    divider = "─" * 60
    print(f"\n{divider}")
    print(f"  📁  All outputs saved to: {output_dir.resolve()}")
    print(f"  📊  Plots: {output_dir / 'plots'}")
    print(f"  🧠  Models: {output_dir / 'models'}")
    print(f"  🔮  Inference: from inference import RULPredictor")
    print(f"{divider}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare RUL prediction models.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Directory for saved models and plots (default: outputs/)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum LSTM training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="LSTM batch size (default: 64)",
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM training (train only RF + XGBoost)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    args = _parse_args()
    train_all(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_lstm=args.skip_lstm,
    )


if __name__ == "__main__":
    main()
