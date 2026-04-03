"""IC-weighted composite model wrapper.

For each rolling window, computes the rank IC (Spearman correlation between
each factor and forward returns) over the training period, then uses these
ICs as weights to combine factors in the test period.

Negative ICs are clipped to 0 (factor gets zero weight if it was
anti-predictive in the training window).
"""

import numpy as np
from ..base import ModelWrapper


class ICWeightModel(ModelWrapper):
    """IC-weighted factor combination.

    For each factor f:
        IC_f = mean of daily cross-sectional Spearman(rank_f, forward_return)
               over the training period

    Composite = sum(IC_f * rank_f) / sum(IC_f)

    Args:
        clip_negative: if True, clip negative ICs to 0 (default True)
    """

    def __init__(self, clip_negative: bool = True):
        self.clip_negative = clip_negative
        self.weights_ = None

    def _rank_cross_section(self, arr_2d: np.ndarray) -> np.ndarray:
        """Rank each row (cross-section) of a (T*N_valid,) reshaped or (T, N) array.
        Actually we need per-day ranking. arr_2d: (T, N) -> (T, N) ranks."""
        T, N = arr_2d.shape
        result = np.full_like(arr_2d, np.nan)
        # Loop: ~T iterations (training period ~504 days)
        for t in range(T):
            row = arr_2d[t]
            valid = np.isfinite(row)
            n_valid = valid.sum()
            if n_valid < 2:
                continue
            order = np.argsort(row[valid])
            ranks = np.empty(n_valid, dtype=np.float64)
            ranks[order] = np.arange(n_valid, dtype=np.float64)
            ranks /= (n_valid - 1)
            result[t, valid] = ranks
        return result

    def _compute_daily_ic(self, factor_2d: np.ndarray, label_2d: np.ndarray) -> float:
        """Compute mean daily rank IC between factor and label.
        factor_2d, label_2d: (T, N). Returns scalar mean IC."""
        T, N = factor_2d.shape
        ics = []
        # Loop: ~T iterations (training period)
        for t in range(T):
            f_row = factor_2d[t]
            l_row = label_2d[t]
            valid = np.isfinite(f_row) & np.isfinite(l_row)
            n = valid.sum()
            if n < 10:
                continue
            # Spearman = Pearson of ranks
            f_ranks = np.argsort(np.argsort(f_row[valid])).astype(np.float64)
            l_ranks = np.argsort(np.argsort(l_row[valid])).astype(np.float64)
            f_ranks -= f_ranks.mean()
            l_ranks -= l_ranks.mean()
            denom = np.sqrt((f_ranks ** 2).sum() * (l_ranks ** 2).sum())
            if denom < 1e-12:
                continue
            ic = (f_ranks * l_ranks).sum() / denom
            ics.append(ic)
        return np.mean(ics) if ics else 0.0

    def fit(self,
            train_X: np.ndarray,   # (T_train, N, F)
            train_y: np.ndarray,   # (T_train, N)
            val_X: np.ndarray,
            val_y: np.ndarray) -> None:
        T, N, F = train_X.shape
        ics = np.zeros(F)

        # Loop: ~F iterations (5 factors)
        for f in range(F):
            ics[f] = self._compute_daily_ic(train_X[:, :, f], train_y)

        if self.clip_negative:
            ics = np.maximum(ics, 0.0)

        total = ics.sum()
        if total < 1e-12:
            # Fallback to equal weight
            self.weights_ = np.ones(F) / F
        else:
            self.weights_ = ics / total

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        # test_X: (T_test, N, F)
        # Weighted sum across factors
        # output: (T_test, N)
        return np.nansum(test_X * self.weights_[np.newaxis, np.newaxis, :], axis=2)
