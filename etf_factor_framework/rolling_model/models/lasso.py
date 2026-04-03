"""Lasso regression model wrapper."""

import numpy as np
from sklearn.linear_model import Lasso
from ..base import ModelWrapper


class LassoModel(ModelWrapper):
    """Lasso (L1-regularized) linear regression.

    Flattens cross-section to (T*N, F), drops NaN rows.
    Validation set is used for alpha selection if alpha_candidates is provided,
    otherwise uses fixed alpha.

    Args:
        alpha: regularization strength (default 0.01)
        alpha_candidates: list of alphas to try, picks best on validation set
        max_iter: maximum iterations (default 5000)
    """

    def __init__(self,
                 alpha: float = 0.01,
                 alpha_candidates: list = None,
                 max_iter: int = 5000):
        self.alpha = alpha
        self.alpha_candidates = alpha_candidates
        self.max_iter = max_iter
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

        if len(y_train) < F + 1:
            self.model_ = None
            return

        if self.alpha_candidates is not None:
            # Select alpha using validation set
            X_val, y_val = self._flatten_and_clean(val_X, val_y)
            best_alpha = self.alpha
            best_score = -np.inf

            for a in self.alpha_candidates:
                m = Lasso(alpha=a, max_iter=self.max_iter)
                m.fit(X_train, y_train)
                if len(y_val) > 0:
                    score = m.score(X_val, y_val)  # R^2
                    if score > best_score:
                        best_score = score
                        best_alpha = a

            self.model_ = Lasso(alpha=best_alpha, max_iter=self.max_iter)
            self.model_.fit(X_train, y_train)
        else:
            self.model_ = Lasso(alpha=self.alpha, max_iter=self.max_iter)
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
