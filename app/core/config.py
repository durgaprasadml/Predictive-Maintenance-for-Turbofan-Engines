"""
Configuration module for the C-MAPSS Turbofan RUL Prediction Pipeline.

Centralizes all tunable hyperparameters, file paths, and column definitions
so that downstream modules remain free of magic numbers and hardcoded values.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration for the C-MAPSS FD001 data pipeline."""

    # ── File Paths ──────────────────────────────────────────────────────
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )
    train_file: str = "train_FD001.txt"
    test_file: str = "test_FD001.txt"
    rul_file: str = "RUL_FD001.txt"

    # ── Column Schema ───────────────────────────────────────────────────
    index_cols: List[str] = field(
        default_factory=lambda: ["unit_number", "time_cycles"]
    )
    setting_cols: List[str] = field(
        default_factory=lambda: ["setting_1", "setting_2", "setting_3"]
    )
    sensor_cols: List[str] = field(
        default_factory=lambda: [f"s_{i}" for i in range(1, 22)]
    )

    # ── Preprocessing ───────────────────────────────────────────────────
    variance_threshold: float = 0.0001  # sensors below this are dropped
    rul_clip_value: int = 125           # piecewise-linear RUL cap (None to disable)

    # ── Sequence Generation ─────────────────────────────────────────────
    sequence_length: int = 30           # sliding window size (cycles)

    # ── Train / Validation Split ────────────────────────────────────────
    test_size: float = 0.2
    random_state: int = 42

    # ── Derived ─────────────────────────────────────────────────────────
    @property
    def all_cols(self) -> List[str]:
        """Full ordered column list matching the raw text file layout."""
        return self.index_cols + self.setting_cols + self.sensor_cols

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file

    @property
    def rul_path(self) -> Path:
        return self.data_dir / self.rul_file
