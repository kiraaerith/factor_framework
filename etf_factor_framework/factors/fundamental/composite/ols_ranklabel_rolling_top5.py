"""
OLS_RANKLABEL_ROLLING_TOP5: OLS rolling composite with ranked labels.

Same as OLS_ROLLING_TOP5 but uses cross-sectional rank of forward returns
as labels instead of raw returns. This reduces label noise and focuses
the regression on relative stock ordering rather than absolute returns.

Factor direction: positive
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from rolling_model.models.ols import OLSModel
from factors.fundamental.composite.rolling_base import RollingCompositeBase


class OLS_RANKLABEL_ROLLING_TOP5(RollingCompositeBase):
    """OLS rolling composite with cross-sectional ranked labels."""

    FACTOR_NAME = "OLS_RANKLABEL_ROLLING_TOP5"
    FACTOR_DIRECTION = 1
    MODEL_CLASS = OLSModel
    MODEL_PARAMS = {"fit_intercept": True}
    RANK_LABELS = True
    TRAINER_PARAMS = {
        "train_min_days": 504,
        "train_max_days": 1260,
        "val_days": 126,
        "test_days": 126,
        "step_days": 126,
    }
