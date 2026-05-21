"""
data_pipeline.py — Production-ready C-MAPSS FD001 preprocessing pipeline.

Responsibilities
────────────────
1. Load raw space-separated text files (no headers).
2. Compute Remaining Useful Life (RUL) per row.
3. Optionally clip RUL with a piecewise-linear cap.
4. Remove non-informative (constant / near-zero variance) sensors.
5. Normalize sensor readings with MinMaxScaler.
6. Generate fixed-length sliding-window sequences for LSTM consumption.
7. Produce train / validation splits at the *engine* level.

Public API
──────────
    load_dataset        → raw DataFrame
    add_rul             → DataFrame with RUL column
    remove_low_variance → DataFrame with uninformative sensors dropped
    normalize_sensors   → (DataFrame, fitted MinMaxScaler)
    build_sequences     → (X: np.ndarray, y: np.ndarray)
    run_pipeline        → full end-to-end bundle (PipelineResult)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from app.core.config import DatasetConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Result Container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Immutable container returned by `run_pipeline`."""

    # Cleaned tabular data
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    # Sequence arrays ready for LSTM (samples, timesteps, features)
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    # Test sequences (one per engine, last window)
    X_test: np.ndarray
    y_test: np.ndarray

    # Fitted scaler for inference-time reuse
    scaler: MinMaxScaler

    # Columns retained after variance filtering
    feature_cols: List[str]

    # Sensors that were dropped
    dropped_sensors: List[str]


# ═══════════════════════════════════════════════════════════════════════
# 1. Loading
# ═══════════════════════════════════════════════════════════════════════

def load_dataset(filepath: Path, columns: List[str]) -> pd.DataFrame:
    """
    Load a C-MAPSS text file into a DataFrame.

    Parameters
    ----------
    filepath : Path
        Absolute or relative path to the `.txt` file.
    columns : list[str]
        Ordered column names matching the file layout.

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath.resolve()}'. "
            f"Download FD001 from the NASA Prognostics Data Repository "
            f"and place the files in '{filepath.parent.resolve()}'."
        )

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=columns,
        engine="python",
    )
    logger.info(
        "Loaded %s → %d rows × %d cols",
        filepath.name, len(df), len(df.columns),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. RUL Computation
# ═══════════════════════════════════════════════════════════════════════

def add_rul(
    df: pd.DataFrame,
    clip_value: Optional[int] = 125,
) -> pd.DataFrame:
    """
    Compute Remaining Useful Life for every row.

        RUL_i = max_cycle(engine) − current_cycle_i

    Optionally applies a piecewise-linear cap so the model focuses on
    the degradation phase rather than memorising arbitrarily large RUL
    values for healthy engines.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `unit_number` and `time_cycles`.
    clip_value : int | None
        If set, RUL is clipped to this maximum.

    Returns
    -------
    pd.DataFrame  (new copy with `RUL` column appended)
    """
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_cycles"]

    if clip_value is not None:
        df["RUL"] = df["RUL"].clip(upper=clip_value)
        logger.info("RUL clipped to [0, %d]", clip_value)

    logger.info(
        "RUL stats → min=%d, median=%d, max=%d",
        df["RUL"].min(), df["RUL"].median(), df["RUL"].max(),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. Variance Filtering
# ═══════════════════════════════════════════════════════════════════════

def remove_low_variance(
    df: pd.DataFrame,
    sensor_cols: List[str],
    threshold: float = 0.0001,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Drop sensors whose variance falls below `threshold`.

    Parameters
    ----------
    df : pd.DataFrame
    sensor_cols : list[str]
        Candidate sensor columns.
    threshold : float
        Minimum acceptable variance.

    Returns
    -------
    (filtered_df, retained_sensors, dropped_sensors)
    """
    variances = df[sensor_cols].var()
    low_var_mask = variances < threshold
    dropped = variances[low_var_mask].index.tolist()
    retained = variances[~low_var_mask].index.tolist()

    df = df.drop(columns=dropped)

    logger.info(
        "Variance filter (threshold=%.4f): kept %d / %d sensors. "
        "Dropped: %s",
        threshold, len(retained), len(sensor_cols),
        dropped if dropped else "none",
    )
    return df, retained, dropped


# ═══════════════════════════════════════════════════════════════════════
# 4. Normalisation
# ═══════════════════════════════════════════════════════════════════════

def normalize_sensors(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], MinMaxScaler]:
    """
    Fit a MinMaxScaler on training data and transform both splits.

    Parameters
    ----------
    train_df : pd.DataFrame
    feature_cols : list[str]
        Columns to scale (sensors + settings).
    test_df : pd.DataFrame | None
        Optional test split to transform with the *same* scaler.

    Returns
    -------
    (scaled_train_df, scaled_test_df | None, fitted_scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_df = train_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

    scaled_test = None
    if test_df is not None:
        scaled_test = test_df.copy()
        scaled_test[feature_cols] = scaler.transform(test_df[feature_cols])

    logger.info("MinMaxScaler fitted on %d features.", len(feature_cols))
    return train_df, scaled_test, scaler


# ═══════════════════════════════════════════════════════════════════════
# 5. Sequence Generation
# ═══════════════════════════════════════════════════════════════════════

def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
    target_col: str = "RUL",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences per engine for LSTM input.

    For each engine, a window of `sequence_length` consecutive cycles is
    slid one step at a time.  Engines with fewer cycles than the window
    are zero-padded on the left.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `unit_number`, `feature_cols`, and `target_col`.
    feature_cols : list[str]
    sequence_length : int
    target_col : str

    Returns
    -------
    (X, y)
        X : np.ndarray of shape (n_samples, sequence_length, n_features)
        y : np.ndarray of shape (n_samples,)
    """
    sequences: List[np.ndarray] = []
    labels: List[float] = []

    for unit_id, group in df.groupby("unit_number"):
        features = group[feature_cols].values  # (T, F)
        target = group[target_col].values       # (T,)
        n_cycles = len(features)

        if n_cycles >= sequence_length:
            # Normal sliding window
            for i in range(n_cycles - sequence_length + 1):
                sequences.append(features[i : i + sequence_length])
                labels.append(target[i + sequence_length - 1])
        else:
            # Zero-pad short engines (pad on the left)
            pad_width = sequence_length - n_cycles
            padded = np.pad(
                features,
                ((pad_width, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            sequences.append(padded)
            labels.append(target[-1])

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    logger.info(
        "Sequences built → X%s, y%s", X.shape, y.shape,
    )
    return X, y


def build_test_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
) -> np.ndarray:
    """
    Build *one* sequence per engine from the test set — the last
    `sequence_length` cycles — mirroring evaluation protocol.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    sequence_length : int

    Returns
    -------
    np.ndarray of shape (n_engines, sequence_length, n_features)
    """
    sequences: List[np.ndarray] = []

    for _, group in df.groupby("unit_number"):
        features = group[feature_cols].values
        n_cycles = len(features)

        if n_cycles >= sequence_length:
            sequences.append(features[-sequence_length:])
        else:
            pad_width = sequence_length - n_cycles
            padded = np.pad(
                features,
                ((pad_width, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            sequences.append(padded)

    X = np.array(sequences, dtype=np.float32)
    logger.info("Test sequences → X%s", X.shape)
    return X


# ═══════════════════════════════════════════════════════════════════════
# 6. Engine-level Train / Validation Split
# ═══════════════════════════════════════════════════════════════════════

def split_by_engine(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences so that no engine appears in both train and
    validation sets (prevents data leakage).

    Parameters
    ----------
    X, y : np.ndarray
        Sequence arrays produced by `build_sequences`.
    groups : np.ndarray
        Engine id for each sample (same length as X).
    test_size : float
    random_state : int

    Returns
    -------
    (X_train, X_val, y_train, y_val)
    """
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state,
    )
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    logger.info(
        "Engine-level split → train %d samples, val %d samples",
        len(train_idx), len(val_idx),
    )
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# ═══════════════════════════════════════════════════════════════════════
# Helper: map each sequence back to its engine id
# ═══════════════════════════════════════════════════════════════════════

def _sequence_engine_ids(
    df: pd.DataFrame,
    sequence_length: int,
) -> np.ndarray:
    """Return an array of engine ids aligned with `build_sequences` output."""
    ids: List[int] = []
    for unit_id, group in df.groupby("unit_number"):
        n = len(group)
        n_seqs = max(n - sequence_length + 1, 1)
        ids.extend([unit_id] * n_seqs)
    return np.array(ids)


# ═══════════════════════════════════════════════════════════════════════
# 7. End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: Optional[DatasetConfig] = None) -> PipelineResult:
    """
    Execute the full preprocessing pipeline and return a
    `PipelineResult` ready for model training.

    Parameters
    ----------
    cfg : DatasetConfig | None
        If None, uses default configuration.

    Returns
    -------
    PipelineResult
    """
    if cfg is None:
        cfg = DatasetConfig()

    # ── 1. Load ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/6 — Loading datasets")
    logger.info("=" * 60)
    train_df = load_dataset(cfg.train_path, cfg.all_cols)
    test_df = load_dataset(cfg.test_path, cfg.all_cols)
    rul_df = pd.read_csv(
        cfg.rul_path, sep=r"\s+", header=None, names=["RUL"],
    )

    # ── 2. Compute RUL ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/6 — Computing RUL")
    logger.info("=" * 60)
    train_df = add_rul(train_df, clip_value=cfg.rul_clip_value)

    # For the test set, ground-truth RUL is provided per engine
    # (the remaining life at the *last* cycle in each engine's test data).
    # We reconstruct per-row RUL from those ground-truth values.
    test_df = _add_test_rul(test_df, rul_df, clip_value=cfg.rul_clip_value)

    # ── 3. Variance filter ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3/6 — Removing low-variance sensors")
    logger.info("=" * 60)
    train_df, retained_sensors, dropped_sensors = remove_low_variance(
        train_df, cfg.sensor_cols, threshold=cfg.variance_threshold,
    )
    # Apply same columns to test
    test_df = test_df.drop(columns=dropped_sensors, errors="ignore")

    feature_cols = cfg.setting_cols + retained_sensors

    # ── 4. Normalise ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4/6 — Normalizing features")
    logger.info("=" * 60)
    train_df, test_df, scaler = normalize_sensors(
        train_df, feature_cols, test_df,
    )

    # ── 5. Sequences ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5/6 — Building sequences (window=%d)", cfg.sequence_length)
    logger.info("=" * 60)
    X_train_full, y_train_full = build_sequences(
        train_df, feature_cols, cfg.sequence_length,
    )
    X_test = build_test_sequences(
        test_df, feature_cols, cfg.sequence_length,
    )
    y_test = rul_df["RUL"].values.astype(np.float32)
    if cfg.rul_clip_value is not None:
        y_test = np.clip(y_test, 0, cfg.rul_clip_value)

    # ── 6. Train / Validation split ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6/6 — Splitting train/validation by engine")
    logger.info("=" * 60)
    engine_ids = _sequence_engine_ids(train_df, cfg.sequence_length)
    X_train, X_val, y_train, y_val = split_by_engine(
        X_train_full, y_train_full, engine_ids,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete ✓")
    logger.info("=" * 60)

    return PipelineResult(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_cols=feature_cols,
        dropped_sensors=dropped_sensors,
    )


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _add_test_rul(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    clip_value: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reconstruct per-row RUL for the test set.

    The ground-truth file gives the RUL at the *last* cycle of each
    engine.  Per-row RUL is:  rul_gt + (max_cycle − current_cycle).
    """
    test_df = test_df.copy()
    max_cycles = test_df.groupby("unit_number")["time_cycles"].transform("max")
    # Map each engine to its ground-truth RUL at the last cycle
    engine_rul = dict(
        zip(
            test_df["unit_number"].unique(),
            rul_df["RUL"].values,
        )
    )
    gt_rul = test_df["unit_number"].map(engine_rul)
    test_df["RUL"] = gt_rul + (max_cycles - test_df["time_cycles"])

    if clip_value is not None:
        test_df["RUL"] = test_df["RUL"].clip(upper=clip_value)

    return test_df
