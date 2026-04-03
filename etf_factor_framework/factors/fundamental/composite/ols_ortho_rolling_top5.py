"""
OLS_ORTHO_ROLLING_TOP5: OLS with symmetric orthogonalization, rolling composite.

Applies symmetric orthogonalization (Σ^{-1/2}) to factor ranks before OLS,
eliminating multicollinearity (especially SIZE-RP_EP correlation ~0.74).

Factor direction: positive
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from rolling_model.models.ols_ortho import OLSOrthoModel
from factors.fundamental.composite.rolling_base import RollingCompositeBase


class OLS_ORTHO_ROLLING_TOP5(RollingCompositeBase):
    """OLS + symmetric orthogonalization rolling composite."""

    FACTOR_NAME = "OLS_ORTHO_ROLLING_TOP5"
    FACTOR_DIRECTION = 1
    MODEL_CLASS = OLSOrthoModel
    MODEL_PARAMS = {"fit_intercept": True}
    TRAINER_PARAMS = {
        "train_min_days": 504,
        "train_max_days": 1260,
        "val_days": 126,
        "test_days": 126,
        "step_days": 126,
    }
