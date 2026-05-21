"""
evaluate.py — Evaluation metrics and visualization for RUL models.

Provides:
    compute_metrics   → RMSE, MAE, R² for a set of predictions
    compare_models    → side-by-side metrics table across models
    plot_predictions  → actual vs predicted scatter/line charts
    plot_comparison   → bar chart comparing RMSE & MAE across models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI use
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ── Plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d29",
    "axes.edgecolor": "#2d3250",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0",
    "grid.color": "#2d3250",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 150,
})

PALETTE = {
    "RandomForest": "#00d4aa",
    "XGBoost": "#7c4dff",
    "LSTM": "#ff6b6b",
}


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    """Container for regression evaluation metrics."""
    rmse: float
    mae: float
    r2: float

    def __str__(self) -> str:
        return f"RMSE={self.rmse:.2f}  MAE={self.mae:.2f}  R²={self.r2:.4f}"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute RMSE, MAE, and R² between true and predicted RUL."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(rmse=rmse, mae=mae, r2=r2)


def compare_models(
    results: Dict[str, Metrics],
) -> str:
    """
    Format a comparison table from a dict of {model_name: Metrics}.

    Returns
    -------
    Formatted string table suitable for logging / printing.
    """
    header = f"  {'Model':<18} {'RMSE':>8} {'MAE':>8} {'R²':>10}"
    divider = f"  {'─'*18} {'─'*8} {'─'*8} {'─'*10}"
    rows = [header, divider]
    for name, m in results.items():
        rows.append(
            f"  {name:<18} {m.rmse:>8.2f} {m.mae:>8.2f} {m.r2:>10.4f}"
        )
    return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path,
    metrics: Optional[Metrics] = None,
) -> Path:
    """
    Plot actual vs predicted RUL with a perfect-prediction diagonal.

    Saves the figure as `{model_name}_predictions.png` in output_dir.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    color = PALETTE.get(model_name, "#00d4aa")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: Scatter ───────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, c=color, edgecolors="none")
    lims = [
        min(y_true.min(), y_pred.min()) - 5,
        max(y_true.max(), y_pred.max()) + 5,
    ]
    ax.plot(lims, lims, "--", color="#ffffff", alpha=0.4, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"{model_name} — Actual vs Predicted")
    ax.grid(True, alpha=0.3)

    if metrics:
        stats = f"RMSE: {metrics.rmse:.2f}\nMAE:  {metrics.mae:.2f}\nR²:   {metrics.r2:.4f}"
        ax.text(
            0.05, 0.95, stats,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f1117",
                      edgecolor=color, alpha=0.85),
        )

    # ── Right: Line plot (sorted by actual) ─────────────────────────
    ax = axes[1]
    sort_idx = np.argsort(y_true)
    ax.plot(y_true[sort_idx], label="Actual", color="#ffffff", alpha=0.7, linewidth=1.2)
    ax.plot(y_pred[sort_idx], label="Predicted", color=color, alpha=0.85, linewidth=1.2)
    ax.fill_between(
        range(len(sort_idx)),
        y_true[sort_idx],
        y_pred[sort_idx],
        alpha=0.15,
        color=color,
    )
    ax.set_xlabel("Sample (sorted by actual RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{model_name} — Prediction Overlay")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)
    filepath = output_dir / f"{model_name.lower()}_predictions.png"
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved prediction plot → %s", filepath)
    return filepath


def plot_comparison(
    results: Dict[str, Metrics],
    output_dir: Path,
) -> Path:
    """
    Bar chart comparing RMSE and MAE across all models.

    Saves the figure as `model_comparison.png`.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    rmse_vals = [results[n].rmse for n in names]
    mae_vals = [results[n].mae for n in names]
    colors = [PALETTE.get(n, "#00d4aa") for n in names]

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(
        x - width / 2, rmse_vals, width,
        label="RMSE", color=colors, alpha=0.85, edgecolor="#0f1117",
    )
    bars2 = ax.bar(
        x + width / 2, mae_vals, width,
        label="MAE", color=colors, alpha=0.50, edgecolor="#0f1117",
    )

    # Value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Error (cycles)")
    ax.set_title("Model Comparison — RMSE & MAE", fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(pad=2.0)
    filepath = output_dir / "model_comparison.png"
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved comparison plot → %s", filepath)
    return filepath


def plot_training_history(
    history: dict,
    output_dir: Path,
) -> Path:
    """
    Plot LSTM training loss and MAE across epochs.

    Parameters
    ----------
    history : dict
        Keras History.history dict with keys like 'loss', 'val_loss', 'mae', 'val_mae'.
    output_dir : Path

    Returns
    -------
    Path to the saved figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    color = PALETTE["LSTM"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history["loss"], label="Train Loss", color=color, linewidth=1.5)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss", color="#ffffff",
                alpha=0.7, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Training Loss")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # ── MAE ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(history["mae"], label="Train MAE", color=color, linewidth=1.5)
    if "val_mae" in history:
        ax.plot(history["val_mae"], label="Val MAE", color="#ffffff",
                alpha=0.7, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (cycles)")
    ax.set_title("LSTM Training MAE")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)
    filepath = output_dir / "lstm_training_history.png"
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved training history → %s", filepath)
    return filepath
