"""
RollingTrainer: rolling window orchestrator for model training and inference.

Splits data into rolling train/val/test windows, creates a fresh model instance
per window, trains, predicts, and concatenates test predictions into the final
factor value matrix.
"""

import logging
import time
from typing import List, Optional, Type

import numpy as np

from .base import ModelWrapper, PreparedData
from .diagnostics import DiagnosticsTool

logger = logging.getLogger(__name__)


class RollingTrainer:
    """Rolling window training orchestrator.

    Training window grows from train_min_days to train_max_days, then slides
    with fixed train_max_days length. Each step advances by step_days.

    Args:
        train_min_days: minimum training window length (default 504, ~2 years)
        train_max_days: maximum training window length (default 1260, ~5 years)
        val_days: validation window length (default 126, ~half year)
        test_days: test window length (default 126, ~half year)
        step_days: rolling step size (default 126, ~half year)
        diagnostics_enabled: whether to compute and store diagnostics (default False)
        diagnostics_db_path: path to SQLite database for diagnostics
    """

    def __init__(self,
                 train_min_days: int = 504,
                 train_max_days: int = 1260,
                 val_days: int = 126,
                 test_days: int = 126,
                 step_days: int = 126,
                 diagnostics_enabled: bool = False,
                 diagnostics_db_path: Optional[str] = None):
        self.train_min_days = train_min_days
        self.train_max_days = train_max_days
        self.val_days = val_days
        self.test_days = test_days
        self.step_days = step_days
        self.diagnostics_enabled = diagnostics_enabled
        self.diagnostics_db_path = diagnostics_db_path

        if diagnostics_enabled and not diagnostics_db_path:
            raise ValueError("diagnostics_db_path required when diagnostics_enabled=True")

    def _compute_windows(self, T: int) -> List[dict]:
        """Compute all rolling window index ranges.

        Returns list of dicts with keys:
            train_start, train_end, val_start, val_end, test_start, test_end
        All indices are integer positions into the time axis [0, T).
        train/val/test ranges are [start, end) (end exclusive).
        """
        windows = []
        test_start = self.train_min_days + self.val_days

        while test_start + self.test_days <= T:
            test_end = test_start + self.test_days
            val_start = test_start - self.val_days
            train_end = val_start

            # Training window: grows from min to max, then slides
            train_start = max(0, train_end - self.train_max_days)

            if train_end - train_start < self.train_min_days:
                break

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_start + self.val_days,
                'test_start': test_start,
                'test_end': test_end,
            })

            test_start += self.step_days

        return windows

    def run(self,
            data: PreparedData,
            model_class: Type[ModelWrapper],
            model_params: dict
            ) -> np.ndarray:
        """Run rolling training and return factor value matrix.

        Args:
            data: PreparedData with features (T, N, F), labels (T, N), etc.
            model_class: ModelWrapper subclass to instantiate each window
            model_params: kwargs passed to model_class.__init__

        Returns:
            (T, N) ndarray of factor values. Positions not covered by any
            test window are NaN.
        """
        T, N, F = data.features.shape
        windows = self._compute_windows(T)

        if not windows:
            logger.warning("No valid windows. Check data length vs window parameters.")
            return np.full((T, N), np.nan)

        logger.info(
            f"RollingTrainer: {len(windows)} windows, "
            f"T={T}, N={N}, F={F}, "
            f"train=[{self.train_min_days},{self.train_max_days}], "
            f"val={self.val_days}, test={self.test_days}, step={self.step_days}"
        )

        output = np.full((T, N), np.nan)

        diag = None
        if self.diagnostics_enabled:
            diag = DiagnosticsTool(self.diagnostics_db_path)

        total_start = time.time()

        # Loop: ~len(windows)次 (typically 15-25 for 10yr data with 6m step)
        for i, w in enumerate(windows):
            window_start = time.time()

            ts, te = w['train_start'], w['train_end']
            vs, ve = w['val_start'], w['val_end']
            xs, xe = w['test_start'], w['test_end']

            train_days = te - ts
            train_dates = (str(data.dates[ts]), str(data.dates[te - 1]))
            val_dates = (str(data.dates[vs]), str(data.dates[ve - 1]))
            test_dates = (str(data.dates[xs]), str(data.dates[xe - 1]))

            logger.info(
                f"  Window {i+1}/{len(windows)}: "
                f"train[{train_dates[0]}~{train_dates[1]}]({train_days}d) "
                f"val[{val_dates[0]}~{val_dates[1]}] "
                f"test[{test_dates[0]}~{test_dates[1]}]"
            )

            train_X = data.features[ts:te]   # (T_train, N, F)
            train_y = data.labels[ts:te]     # (T_train, N)
            val_X = data.features[vs:ve]     # (T_val, N, F)
            val_y = data.labels[vs:ve]       # (T_val, N)
            test_X = data.features[xs:xe]    # (T_test, N, F)

            # Create fresh model instance
            model = model_class(**model_params)

            # Train
            model.fit(train_X, train_y, val_X, val_y)

            # Predict on test set
            test_pred = model.predict(test_X)  # (T_test, N)
            output[xs:xe] = test_pred

            # Diagnostics (optional)
            if diag is not None:
                train_pred = model.predict(train_X)
                val_pred = model.predict(val_X)

                window_info = {
                    'train_start': train_dates[0],
                    'train_end': train_dates[1],
                    'val_start': val_dates[0],
                    'val_end': val_dates[1],
                }
                diag.compute_and_save(i, window_info,
                                      train_pred, train_y,
                                      val_pred, val_y)
                del train_pred, val_pred

            del model, train_X, train_y, val_X, val_y, test_X, test_pred

            elapsed = time.time() - window_start
            logger.info(f"    Done in {elapsed:.1f}s")

        if diag is not None:
            diag.close()

        total_elapsed = time.time() - total_start
        # Count non-NaN coverage
        coverage = np.isfinite(output).any(axis=1).sum()
        logger.info(
            f"RollingTrainer complete: {len(windows)} windows in {total_elapsed:.1f}s, "
            f"output coverage: {coverage}/{T} days"
        )

        return output
