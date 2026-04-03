"""
OLS rolling composite of 18-factor pool (cached version).

Method: Rolling OLS regression on cross-sectional ranks -> forward 20d returns.
Train: 2-5yr expanding, val: 6m, test: 6m, step: 6m.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from rolling_model.models.ols import OLSModel
from factors.fundamental.composite.rolling_base_cached import RollingCompositeCached


class OLS_ROLLING_POOL18_CACHED(RollingCompositeCached):
    """OLS rolling composite, 18 factors, raw labels."""
    FACTOR_NAME = "OLS_ROLLING_POOL18_CACHED"
    MODEL_CLASS = OLSModel
    MODEL_PARAMS = {"fit_intercept": True}
    RANK_LABELS = False
    TRAINER_PARAMS = {
        "train_min_days": 504,
        "train_max_days": 1260,
        "val_days": 126,
        "test_days": 126,
        "step_days": 126,
    }


class OLS_RANKLABEL_POOL18_CACHED(RollingCompositeCached):
    """OLS rolling composite, 18 factors, rank labels."""
    FACTOR_NAME = "OLS_RANKLABEL_POOL18_CACHED"
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


class OLS_ZSCORE_POOL18_CACHED(RollingCompositeCached):
    """OLS rolling composite, 18 factors, zscore preprocessing, raw labels."""
    FACTOR_NAME = "OLS_ZSCORE_POOL18_CACHED"
    MODEL_CLASS = OLSModel
    MODEL_PARAMS = {"fit_intercept": True}
    RANK_LABELS = False
    PREPROCESS_STEPS = ['zscore']
    TRAINER_PARAMS = {
        "train_min_days": 504,
        "train_max_days": 1260,
        "val_days": 126,
        "test_days": 126,
        "step_days": 126,
    }


class OLS_ZSCORE_RANKLABEL_POOL18_CACHED(RollingCompositeCached):
    """OLS rolling composite, 18 factors, zscore preprocessing, rank labels."""
    FACTOR_NAME = "OLS_ZSCORE_RANKLABEL_POOL18_CACHED"
    MODEL_CLASS = OLSModel
    MODEL_PARAMS = {"fit_intercept": True}
    RANK_LABELS = True
    PREPROCESS_STEPS = ['zscore']
    TRAINER_PARAMS = {
        "train_min_days": 504,
        "train_max_days": 1260,
        "val_days": 126,
        "test_days": 126,
        "step_days": 126,
    }
