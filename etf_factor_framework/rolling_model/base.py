"""
Base interfaces for the rolling model training framework.

- PreparedData: data container for aligned features, labels, dates, symbols
- DataPreparer: abstract interface for loading and aligning data
- ModelWrapper: abstract interface for model training and inference
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PreparedData:
    """Aligned data container for rolling model training.

    Attributes:
        features: (T, N, F) array - T trading days, N stocks, F features
        labels: (T, N) array - target values (e.g. forward returns)
        dates: (T,) array - trading dates
        symbols: (N,) array - stock symbols
        feature_names: list of F feature names
    """
    features: np.ndarray       # (T, N, F)
    labels: np.ndarray         # (T, N)
    dates: np.ndarray          # (T,)
    symbols: np.ndarray        # (N,)
    feature_names: List[str]   # (F,)

    def __post_init__(self):
        T, N, F = self.features.shape
        assert self.labels.shape == (T, N), \
            f"labels shape {self.labels.shape} != features (T,N)=({T},{N})"
        assert self.dates.shape == (T,), \
            f"dates length {self.dates.shape[0]} != T={T}"
        assert self.symbols.shape == (N,), \
            f"symbols length {self.symbols.shape[0]} != N={N}"
        assert len(self.feature_names) == F, \
            f"feature_names length {len(self.feature_names)} != F={F}"


class DataPreparer(ABC):
    """Abstract interface for data preparation.

    Subclasses decide how to load and align data (config-driven, hardcoded
    file reads, direct array passthrough, etc.).
    """

    @abstractmethod
    def prepare(self) -> PreparedData:
        """Load and align data, return PreparedData."""


class ModelWrapper(ABC):
    """Abstract interface for model training and inference.

    Each model decides independently:
    - How to reshape data (e.g. flatten cross-section to (T*N, F))
    - How to use the validation set (early stopping, hyperparam selection, etc.)
    - How to handle NaN (drop, fill, native support)
    - Whether to train one model for all stocks or per-stock models

    A new instance is created for each rolling window to avoid state leakage.
    """

    @abstractmethod
    def fit(self,
            train_X: np.ndarray,   # (T_train, N, F)
            train_y: np.ndarray,   # (T_train, N)
            val_X: np.ndarray,     # (T_val, N, F)
            val_y: np.ndarray      # (T_val, N)
            ) -> None:
        """Train the model on training data, using validation data as needed."""

    @abstractmethod
    def predict(self,
                test_X: np.ndarray   # (T_test, N, F)
                ) -> np.ndarray:     # (T_test, N)
        """Generate predictions for test data."""
