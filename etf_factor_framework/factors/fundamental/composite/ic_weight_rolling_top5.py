"""
IC_WEIGHT_ROLLING_TOP5: IC-weighted rolling composite of top 5 factors.

Each rolling window computes rank IC of each factor over the past 2 years,
then uses these ICs as weights to combine factors in the test period.

Factor direction: positive (higher composite = better expected return)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from rolling_model.models.ic_weight import ICWeightModel
from factors.fundamental.composite.rolling_base import RollingCompositeBase


class IC_WEIGHT_ROLLING_TOP5(RollingCompositeBase):
    """IC-weighted rolling composite of 5 top fundamental factors."""

    FACTOR_NAME = "IC_WEIGHT_ROLLING_TOP5"
    FACTOR_DIRECTION = 1
    MODEL_CLASS = ICWeightModel
    MODEL_PARAMS = {"clip_negative": True}
    TRAINER_PARAMS = {
        "train_min_days": 504,    # ~2 years
        "train_max_days": 504,    # fixed 2-year window (IC computed over past 2yr)
        "val_days": 1,            # IC model doesn't use validation set
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
    print("IC_WEIGHT_ROLLING_TOP5 smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2020-01-01", end_date="2024-12-31")
    calc = IC_WEIGHT_ROLLING_TOP5()
    result = calc.calculate(fd)

    print(f"Shape: {result.shape}")
    print(f"NaN ratio: {np.isnan(result.values).mean():.1%}")
    print("[PASS]")
