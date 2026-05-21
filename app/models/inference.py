"""
inference.py — Production inference API for RUL prediction.

Provides a single entry point `predict_rul()` that loads the best saved
model and returns a RUL estimate for a given sensor input sequence.

Usage
─────
    from inference import RULPredictor

    predictor = RULPredictor.load("outputs/models")
    rul = predictor.predict_rul(sensor_sequence)     # np.ndarray (30, 18)
    batch = predictor.predict_batch(sequences)       # np.ndarray (N, 30, 18)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "RandomForest": {"file": "random_forest.joblib", "type": "sklearn"},
    "XGBoost": {"file": "xgboost.joblib", "type": "sklearn"},
    "LSTM": {"file": "lstm_rul.h5", "type": "keras"},
}


class RULPredictor:
    """
    Unified inference wrapper for all RUL model types.

    Handles loading, input validation, reshaping, and prediction
    for both tree-based and LSTM models transparently.
    """

    def __init__(
        self,
        model,
        model_type: str,
        model_name: str,
        expected_shape: tuple,
    ):
        self._model = model
        self._model_type = model_type
        self._model_name = model_name
        self._expected_shape = expected_shape  # (timesteps, features)

    # ── Factory ─────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_dir: Union[str, Path],
        model_name: Optional[str] = None,
    ) -> "RULPredictor":
        """
        Load a saved model for inference.

        Parameters
        ----------
        model_dir : str | Path
            Directory containing saved model files and metadata.json.
        model_name : str | None
            Which model to load: 'RandomForest', 'XGBoost', or 'LSTM'.
            If None, loads the best model recorded in metadata.json.

        Returns
        -------
        RULPredictor instance ready for inference.
        """
        model_dir = Path(model_dir)

        # Determine which model to load
        if model_name is None:
            meta_path = model_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                model_name = meta.get("best_model", "LSTM")
                logger.info("Auto-selected best model: %s", model_name)
            else:
                model_name = "LSTM"
                logger.warning("No metadata.json found; defaulting to LSTM")

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}"
            )

        entry = MODEL_REGISTRY[model_name]
        filepath = model_dir / entry["file"]

        if not filepath.exists():
            raise FileNotFoundError(
                f"Model file not found: {filepath}. "
                f"Run train_models.py first."
            )

        if entry["type"] == "sklearn":
            model = joblib.load(filepath)
            logger.info("Loaded %s from %s", model_name, filepath)
        else:
            import tensorflow as tf
            model = tf.keras.models.load_model(filepath, compile=False)
            logger.info("Loaded %s from %s", model_name, filepath)

        # Load expected input shape from metadata
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            expected_shape = tuple(meta.get("input_shape", [30, 18]))
        else:
            expected_shape = (30, 18)

        return cls(
            model=model,
            model_type=entry["type"],
            model_name=model_name,
            expected_shape=expected_shape,
        )

    # ── Prediction ──────────────────────────────────────────────────

    def predict_rul(self, input_sequence: np.ndarray) -> float:
        """
        Predict RUL for a single sensor sequence.

        Parameters
        ----------
        input_sequence : np.ndarray
            Shape: (timesteps, features) — e.g., (30, 18).
            Must be pre-normalized using the pipeline's fitted scaler.

        Returns
        -------
        float — Predicted Remaining Useful Life in cycles.
        """
        input_sequence = np.asarray(input_sequence, dtype=np.float32)
        self._validate_input(input_sequence, single=True)

        if self._model_type == "sklearn":
            # Flatten for tree models: (1, timesteps * features)
            flat = input_sequence.reshape(1, -1)
            pred = self._model.predict(flat)[0]
        else:
            # LSTM expects (1, timesteps, features)
            batch = input_sequence[np.newaxis, ...]
            pred = float(self._model.predict(batch, verbose=0)[0, 0])

        return max(0.0, float(pred))

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict RUL for a batch of sequences.

        Parameters
        ----------
        sequences : np.ndarray
            Shape: (n_samples, timesteps, features).

        Returns
        -------
        np.ndarray of shape (n_samples,) — Predicted RUL values.
        """
        sequences = np.asarray(sequences, dtype=np.float32)
        self._validate_input(sequences, single=False)

        if self._model_type == "sklearn":
            flat = sequences.reshape(sequences.shape[0], -1)
            preds = self._model.predict(flat)
        else:
            preds = self._model.predict(sequences, verbose=0).flatten()

        return np.clip(preds, 0, None).astype(np.float32)

    # ── Validation ──────────────────────────────────────────────────

    def _validate_input(self, x: np.ndarray, single: bool) -> None:
        """Check input dimensions match the trained model."""
        ts, feat = self._expected_shape
        if single:
            if x.ndim != 2 or x.shape != (ts, feat):
                raise ValueError(
                    f"Expected input shape ({ts}, {feat}), "
                    f"got {x.shape}. Ensure the sequence is "
                    f"pre-normalized and has the correct window size."
                )
        else:
            if x.ndim != 3 or x.shape[1:] != (ts, feat):
                raise ValueError(
                    f"Expected batch shape (N, {ts}, {feat}), "
                    f"got {x.shape}."
                )

    # ── Info ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._model_name

    def __repr__(self) -> str:
        return (
            f"RULPredictor(model={self._model_name}, "
            f"type={self._model_type}, "
            f"input={self._expected_shape})"
        )
