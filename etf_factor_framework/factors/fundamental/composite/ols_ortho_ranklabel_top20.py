"""
OLS_ORTHO_RL_TOP20: OLS + symmetric orthogonalization + rank labels,
rolling composite of top 20 fundamental factors.

Factor direction: positive
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from rolling_model.models.ols_ortho import OLSOrthoModel
from factors.fundamental.composite.rolling_base import RollingCompositeBase


TOP20_SUB_FACTORS = [
    ("factors.fundamental.value.dyr", "DYR", 1),
    ("factors.fundamental.valuation.valuation_SIZE", "valuation_SIZE", -1),
    ("factors.fundamental.growth.ni_st_ft", "NI_ST_FT", 1),
    ("factors.fundamental.value.rp_ep", "RP_EP", 1),
    ("factors.fundamental.growth.na_growth_comp", "NA_GROWTH_COMP", 1),
    ("factors.fundamental.growth.ni_lt_ft", "NI_LT_FT", 1),
    ("factors.fundamental.growth.ni_st_ft_s", "NI_ST_FT_S", 1),
    ("factors.fundamental.value.dpfwd", "DPFWD", 1),
    ("factors.fundamental.growth.ta_growth_comp", "TA_GROWTH_COMP", 1),
    ("factors.fundamental.value.sp_ttm", "SP_TTM", 1),
    ("factors.fundamental.growth.rev_st_ft", "REV_ST_FT", 1),
    ("factors.fundamental.growth.rev_lt_ft", "REV_LT_FT", 1),
    ("factors.fundamental.value.cfp_ttm", "CFP_TTM", 1),
    ("factors.fundamental.growth.rd_to_mv", "RD_TO_MV", 1),
    ("factors.fundamental.value.sev_ttm", "SEV_TTM", 1),
    ("factors.fundamental.growth.rd_growth", "RD_GROWTH", 1),
    ("factors.fundamental.value.cfev_ttm", "CFEV_TTM", 1),
    ("factors.fundamental.growth.comp_robust_growth", "COMP_ROBUST_GROWTH", 1),
    ("factors.fundamental.growth.rd_to_np", "RD_TO_NP", 1),
    ("factors.fundamental.growth.op_growth_comp", "OP_GROWTH_COMP", 1),
]


class OLS_ORTHO_RL_TOP20(RollingCompositeBase):
    """OLS + orthogonalization + rank labels, 20 factors."""

    FACTOR_NAME = "OLS_ORTHO_RL_TOP20"
    FACTOR_DIRECTION = 1
    MODEL_CLASS = OLSOrthoModel
    MODEL_PARAMS = {"fit_intercept": True}
    RANK_LABELS = True
    CUSTOM_SUB_FACTORS = TOP20_SUB_FACTORS
    TRAINER_PARAMS = {
        "train_min_days": 504,
        "train_max_days": 1260,
        "val_days": 126,
        "test_days": 126,
        "step_days": 126,
    }
