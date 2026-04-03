"""
RF_RANKLABEL_ROLLING_TOP5: Random Forest rolling composite with ranked labels.

Factor direction: positive
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from rolling_model.models.random_forest import RandomForestModel
from factors.fundamental.composite.rolling_base import RollingCompositeBase


class RF_RANKLABEL_ROLLING_TOP5(RollingCompositeBase):

    FACTOR_NAME = "RF_RANKLABEL_ROLLING_TOP5"
    FACTOR_DIRECTION = 1
    MODEL_CLASS = RandomForestModel
    MODEL_PARAMS = {
        "n_estimators": 100,
        "max_depth_candidates": [3, 5, 7, 10],
        "min_samples_leaf": 100,
        "max_features": "sqrt",
    }
    RANK_LABELS = True
    TRAINER_PARAMS = {
        "train_min_days": 504,
        "train_max_days": 504,
        "val_days": 126,
        "test_days": 126,
        "step_days": 126,
    }
