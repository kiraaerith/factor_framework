"""OLS (Ordinary Least Squares) model wrapper."""

import numpy as np
from ..base import ModelWrapper


class OLSModel(ModelWrapper):
    """OLS linear regression.

    Flattens cross-section to (T*N, F), drops NaN rows, fits OLS.
    Validation set is not used (OLS has no hyperparameters to tune).

    Args:
        fit_intercept: whether to fit an intercept term (default True)
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self,
            train_X: np.ndarray,
            train_y: np.ndarray,
            val_X: np.ndarray,
            val_y: np.ndarray) -> None:
        T, N, F = train_X.shape
        X = train_X.reshape(-1, F)   # (T*N, F)
        y = train_y.reshape(-1)      # (T*N,)

        # Drop NaN rows
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        if len(y) < F + 1:
            self.coef_ = np.zeros(F)
            self.intercept_ = 0.0
            return

        if self.fit_intercept:
            X = np.column_stack([X, np.ones(len(X))])

        # Normal equation: (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.coef_ = np.zeros(F)
            self.intercept_ = 0.0
            return

        if self.fit_intercept:
            self.coef_ = beta[:-1]
            self.intercept_ = beta[-1]
        else:
            self.coef_ = beta
            self.intercept_ = 0.0

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        T, N, F = test_X.shape
        X = test_X.reshape(-1, F)
        pred = X @ self.coef_ + self.intercept_
        # Preserve NaN: where any feature is NaN, output NaN
        nan_mask = ~np.isfinite(X).all(axis=1)
        pred[nan_mask] = np.nan
        return pred.reshape(T, N)
