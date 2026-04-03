"""Random Forest model wrapper for rolling factor composition.

Uses validation set to select optimal max_depth from candidates.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ..base import ModelWrapper


class RandomForestModel(ModelWrapper):
    """Random Forest regression with validation-based depth selection.

    Args:
        n_estimators: number of trees (default 100)
        max_depth_candidates: list of max_depth values to search (default [3,5,7,10])
        min_samples_leaf: minimum samples per leaf (default 100)
        max_features: features per split (default 'sqrt')
        n_jobs: parallelism, default cpu_count - 1
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth_candidates: list = None,
                 min_samples_leaf: int = 100,
                 max_features: str = 'sqrt',
                 n_jobs: int = None):
        self.n_estimators = n_estimators
        self.max_depth_candidates = max_depth_candidates or [3, 5, 7, 10]
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs or max(1, os.cpu_count() - 1)
        self.model_ = None

    def _flatten_and_clean(self, X_3d, y_2d):
        """Flatten (T, N, F) -> (T*N, F) and drop NaN rows."""
        T, N, F = X_3d.shape
        X = X_3d.reshape(-1, F)
        y = y_2d.reshape(-1)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return X[mask], y[mask]

    def fit(self,
            train_X: np.ndarray,
            train_y: np.ndarray,
            val_X: np.ndarray,
            val_y: np.ndarray) -> None:
        F = train_X.shape[2]
        X_train, y_train = self._flatten_and_clean(train_X, train_y)
        X_val, y_val = self._flatten_and_clean(val_X, val_y)

        if len(y_train) < F + 1:
            self.model_ = None
            return

        # Search best max_depth on validation set
        best_depth = self.max_depth_candidates[0]
        best_score = -np.inf

        for depth in self.max_depth_candidates:
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_jobs=self.n_jobs,
                random_state=42,
            )
            rf.fit(X_train, y_train)

            if len(y_val) > 0:
                score = rf.score(X_val, y_val)  # R^2
                if score > best_score:
                    best_score = score
                    best_depth = depth

        # Retrain with best depth on train data
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=best_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=42,
        )
        self.model_.fit(X_train, y_train)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        T, N, F = test_X.shape
        X = test_X.reshape(-1, F)

        if self.model_ is None:
            return np.full((T, N), np.nan)

        pred = self.model_.predict(X)
        nan_mask = ~np.isfinite(X).all(axis=1)
        pred[nan_mask] = np.nan
        return pred.reshape(T, N)
