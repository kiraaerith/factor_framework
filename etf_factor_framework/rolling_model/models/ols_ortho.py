"""OLS with symmetric orthogonalization preprocessing.

Each rolling window:
1. Compute correlation matrix Σ from training features
2. Symmetric orthogonalization: X* = X @ Σ^(-1/2)
3. Fit OLS on orthogonalized features
4. Apply same transform to test features before prediction
"""

import numpy as np
from ..base import ModelWrapper


class OLSOrthoModel(ModelWrapper):
    """OLS with symmetric orthogonalization.

    Args:
        fit_intercept: whether to fit an intercept term (default True)
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.ortho_matrix_ = None  # Σ^(-1/2)

    def _compute_ortho_matrix(self, X_2d: np.ndarray) -> np.ndarray:
        """Compute Σ^(-1/2) for symmetric orthogonalization.
        X_2d: (M, F) clean feature matrix (no NaN).
        Returns: (F, F) transformation matrix.
        """
        # Correlation matrix
        F = X_2d.shape[1]
        corr = np.corrcoef(X_2d.T)  # (F, F)

        # Eigendecomposition: Σ = P D P^T
        eigvals, eigvecs = np.linalg.eigh(corr)

        # Clip small eigenvalues for numerical stability
        eigvals = np.maximum(eigvals, 1e-8)

        # Σ^(-1/2) = P @ diag(1/sqrt(D)) @ P^T
        inv_sqrt_D = np.diag(1.0 / np.sqrt(eigvals))
        ortho = eigvecs @ inv_sqrt_D @ eigvecs.T

        return ortho

    def _flatten_and_clean(self, X_3d, y_2d):
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
        T, N, F = train_X.shape
        X_train, y_train = self._flatten_and_clean(train_X, train_y)

        if len(y_train) < F + 1:
            self.coef_ = np.zeros(F)
            self.intercept_ = 0.0
            self.ortho_matrix_ = np.eye(F)
            return

        # Step 1: compute orthogonalization matrix from training data
        self.ortho_matrix_ = self._compute_ortho_matrix(X_train)

        # Step 2: transform training features
        X_ortho = X_train @ self.ortho_matrix_

        # Step 3: fit OLS
        if self.fit_intercept:
            X_fit = np.column_stack([X_ortho, np.ones(len(X_ortho))])
        else:
            X_fit = X_ortho

        try:
            beta = np.linalg.lstsq(X_fit, y_train, rcond=None)[0]
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

        # Apply same orthogonalization
        X_ortho = X @ self.ortho_matrix_

        pred = X_ortho @ self.coef_ + self.intercept_

        nan_mask = ~np.isfinite(X).all(axis=1)
        pred[nan_mask] = np.nan
        return pred.reshape(T, N)
