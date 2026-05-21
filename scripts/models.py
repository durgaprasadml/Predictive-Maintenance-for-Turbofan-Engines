"""
models.py — Model definitions for RUL prediction.

Provides factory functions that build, compile, and return:
    1. Random Forest Regressor
    2. XGBoost Regressor
    3. LSTM neural network

Each builder accepts hyperparameters and returns a ready-to-train object.
RF / XGBoost operate on flattened sequences; LSTM uses 3-D input directly.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Utility: Flatten 3-D sequences for tree-based models
# ═══════════════════════════════════════════════════════════════════════

def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Reshape (samples, timesteps, features) → (samples, timesteps * features).

    Tree-based models cannot ingest temporal 3-D arrays, so each
    sequence is unrolled into a single feature vector.
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


# ═══════════════════════════════════════════════════════════════════════
# 1. Random Forest
# ═══════════════════════════════════════════════════════════════════════

def build_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Build a Random Forest regressor tuned for RUL prediction.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int | None
        Maximum tree depth (None = unlimited).
    min_samples_split : int
        Minimum samples to split an internal node.
    min_samples_leaf : int
        Minimum samples at a leaf node.
    n_jobs : int
        Parallel workers (-1 = all cores).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    RandomForestRegressor (unfitted)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )
    logger.info(
        "RandomForest built: %d trees, max_depth=%s",
        n_estimators, max_depth,
    )
    return model


# ═══════════════════════════════════════════════════════════════════════
# 2. XGBoost
# ═══════════════════════════════════════════════════════════════════════

def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 7,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> XGBRegressor:
    """
    Build an XGBoost regressor tuned for RUL prediction.

    Returns
    -------
    XGBRegressor (unfitted)
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )
    logger.info(
        "XGBoost built: %d rounds, lr=%.3f, depth=%d",
        n_estimators, learning_rate, max_depth,
    )
    return model


# ═══════════════════════════════════════════════════════════════════════
# 3. LSTM
# ═══════════════════════════════════════════════════════════════════════

def build_lstm(
    input_shape: Tuple[int, int],
    lstm_units: Tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.2,
    dense_units: Tuple[int, ...] = (32, 16),
    learning_rate: float = 1e-3,
) -> "tensorflow.keras.Model":
    """
    Build and compile a stacked-LSTM model for RUL regression.

    Architecture
    ────────────
        Input (timesteps, features)
        → LSTM(64, return_sequences=True) + Dropout
        → LSTM(32) + Dropout
        → Dense(32, relu) + Dropout
        → Dense(16, relu)
        → Dense(1, linear)   ← RUL output

    Parameters
    ----------
    input_shape : (timesteps, features)
    lstm_units : tuple of ints
        Units per LSTM layer (stacked).
    dropout_rate : float
    dense_units : tuple of ints
        Units per Dense head layer.
    learning_rate : float

    Returns
    -------
    Compiled Keras Model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inputs = layers.Input(shape=input_shape, name="sensor_input")

    x = inputs
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            name=f"lstm_{i+1}",
        )(x)
        x = layers.Dropout(dropout_rate, name=f"lstm_dropout_{i+1}")(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        if i == 0:
            x = layers.Dropout(dropout_rate, name="dense_dropout")(x)

    output = layers.Dense(1, activation="linear", name="rul_output")(x)

    model = models.Model(inputs=inputs, outputs=output, name="LSTM_RUL")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info("LSTM built: input_shape=%s, params=%d",
                input_shape, model.count_params())
    return model
