"""
RF_ROLLING_TOP5: Random Forest rolling composite of top 5 factors.

Uses RollingTrainer + RandomForestModel with validation-based max_depth
selection to capture non-linear factor interactions.

Factor direction: positive (higher RF prediction = better expected return)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from rolling_model.models.random_forest import RandomForestModel
from factors.fundamental.composite.rolling_base import RollingCompositeBase


class RF_ROLLING_TOP5(RollingCompositeBase):
    """Random Forest rolling composite of 5 top fundamental factors."""

    FACTOR_NAME = "RF_ROLLING_TOP5"
    FACTOR_DIRECTION = 1
    MODEL_CLASS = RandomForestModel
    MODEL_PARAMS = {
        "n_estimators": 100,
        "max_depth_candidates": [3, 5, 7, 10],
        "min_samples_leaf": 100,
        "max_features": "sqrt",
    }
    TRAINER_PARAMS = {
        "train_min_days": 504,    # ~2 years
        "train_max_days": 504,    # fixed 2-year window
        "val_days": 126,          # ~half year, for depth selection
        "test_days": 126,         # ~half year
        "step_days": 126,         # ~half year step
    }


if __name__ == "__main__":
    import warnings
    import logging
    import numpy as np
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from factors.fundamental.fundamental_data import FundamentalData

    print("=" * 60)
    print("RF_ROLLING_TOP5 smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2020-01-01", end_date="2024-12-31")
    calc = RF_ROLLING_TOP5()
    result = calc.calculate(fd)

    print(f"Shape: {result.shape}")
    print(f"NaN ratio: {np.isnan(result.values).mean():.1%}")
    print("[PASS]")
