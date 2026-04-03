"""
EQUAL_RANK_POOL18: Equal-weight rank composite of 18-factor pool.

Factor pool (3 tiers):
  Strong (Sharpe > 0.5):
    1. DYR              (+1) dividend yield
    2. valuation_SIZE   (-1) small cap
    3. NI_ST_FT         (+1) analyst NI forecast revision
    4. NA_GROWTH_COMP   (+1) net asset growth composite

  New developed:
    5. CCR              (+1) cash coverage ratio
    6. CFOA             (+1) operating cash flow / total assets
    7. ATD              (+1) asset turnover delta
    8. APR_TTM          (-1) accrual profit ratio (lower=better quality)
    9. GPMD             (+1) gross profit margin delta
   10. TOE              (+1) tax / equity
   11. OCFA             (-1) capex / fixed assets (lower=better utilization)
   12. DPR_TTM          (+1) dividend payout ratio
   13. OPMD             (+1) operating profit margin delta

  Weak/reference:
   14. EP_TTM           (+1) earnings-to-price
   15. ROE_SIMPLE       (+1) return on equity
   16. DAD              (-1) debt-to-asset delta (lower=deleveraging)
   17. ROA_SIMPLE       (+1) return on assets
   18. BP_MRQ           (+1) book-to-price

Method:
  - Cross-sectional percentile rank [0,1] per sub-factor per day
  - Flip rank for negative-direction factors
  - Composite = mean of valid ranks (require >= 6 of 18 valid)

Factor direction: positive (higher composite = better)
"""

import os
import sys
import importlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "EQUAL_RANK_POOL18"
FACTOR_DIRECTION = 1

# (module_path, class_name, direction)
SUB_FACTORS = [
    # Strong
    ("factors.fundamental.value.dyr", "DYR", 1),
    ("factors.fundamental.valuation.valuation_SIZE", "valuation_SIZE", -1),
    ("factors.fundamental.growth.ni_st_ft", "NI_ST_FT", 1),
    ("factors.fundamental.growth.na_growth_comp", "NA_GROWTH_COMP", 1),
    # New developed
    ("factors.fundamental.quality.ccr", "CCR", 1),
    ("factors.fundamental.cashflow.cfoa", "CFOA", 1),
    ("factors.fundamental.efficiency.atd", "ATD", 1),
    ("factors.fundamental.quality.apr_ttm", "APR_TTM", -1),
    ("factors.fundamental.efficiency.gpmd", "GPMD", 1),
    ("factors.fundamental.cashflow.toe", "TOE", 1),
    ("factors.fundamental.efficiency.ocfa", "OCFA", -1),
    ("factors.fundamental.governance.dpr_ttm", "DPR_TTM", 1),
    ("factors.fundamental.efficiency.opmd", "OPMD", 1),
    # Weak/reference
    ("factors.fundamental.value.ep_ttm", "EP_TTM", 1),
    ("factors.fundamental.profitability.roe_simple", "ROE_SIMPLE", 1),
    ("factors.fundamental.leverage.dad", "DAD", -1),
    ("factors.fundamental.profitability.roa_simple", "ROA_SIMPLE", 1),
    ("factors.fundamental.value.bp_mrq", "BP_MRQ", 1),
]

MIN_VALID_FACTORS = 6  # require at least 6 of 18 valid


def _cross_sectional_rank(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per column. NaN stays NaN. Result in [0,1].
    arr: (N, T) -> returns (N, T)
    """
    result = np.full_like(arr, np.nan)
    N, T = arr.shape
    # Loop: ~T iterations (~2430 trading days)
    for t in range(T):
        col = arr[:, t]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        if n_valid < 2:
            continue
        order = np.argsort(col[valid])
        ranks = np.empty(order.shape[0], dtype=np.float64)
        ranks[order] = np.arange(order.shape[0], dtype=np.float64)
        ranks /= (n_valid - 1)
        result[valid, t] = ranks
    return result


class EQUAL_RANK_POOL18(FundamentalFactorCalculator):
    """Equal-weight rank composite of the full 18-factor pool."""

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "sub_factors": [sf[1] for sf in SUB_FACTORS],
            "method": "equal_rank",
            "direction": FACTOR_DIRECTION,
            "n_factors": len(SUB_FACTORS),
            "min_valid": MIN_VALID_FACTORS,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        import logging
        logger = logging.getLogger(__name__)

        # Step 1: compute all sub-factors
        sub_results = []
        for mod_path, cls_name, direction in SUB_FACTORS:
            logger.info(f"  Computing sub-factor: {cls_name}")
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            calculator = cls()
            fd = calculator.calculate(fundamental_data)
            sub_results.append((fd, direction, cls_name))

        # Step 2: find common symbols and dates
        common_symbols_set = None
        common_dates_set = None
        for fd, _, name in sub_results:
            syms = set(fd.symbols.tolist())
            dts = set(fd.dates.astype('int64').tolist())
            if common_symbols_set is None:
                common_symbols_set = syms
                common_dates_set = dts
            else:
                common_symbols_set &= syms
                common_dates_set &= dts
            logger.info(f"    {name}: symbols={len(syms)}, dates={len(dts)}, "
                        f"common_sym={len(common_symbols_set)}, common_dt={len(common_dates_set)}")

        common_symbols = np.array(sorted(common_symbols_set))
        common_dates = np.array(sorted(common_dates_set), dtype='datetime64[ns]')
        N = len(common_symbols)
        T = len(common_dates)
        F = len(SUB_FACTORS)

        logger.info(f"{FACTOR_NAME}: common N={N}, T={T}, F={F}")

        if N == 0 or T == 0:
            raise ValueError(f"{FACTOR_NAME}: no common data. N={N}, T={T}")

        # Step 3: rank, flip, accumulate
        rank_sum = np.zeros((N, T), dtype=np.float64)
        valid_count = np.zeros((N, T), dtype=np.float64)

        for fd, direction, cls_name in sub_results:
            sym_to_idx = {s: i for i, s in enumerate(fd.symbols.tolist())}
            date_to_idx = {int(d): i for i, d in enumerate(fd.dates.astype('int64'))}

            sym_indices = np.array([sym_to_idx[s] for s in common_symbols.tolist()])
            date_indices = np.array([date_to_idx[int(d)] for d in common_dates.astype('int64')])
            aligned = fd.values[np.ix_(sym_indices, date_indices)]

            ranked = _cross_sectional_rank(aligned)

            if direction == -1:
                ranked = np.where(np.isnan(ranked), np.nan, 1.0 - ranked)

            valid_mask = ~np.isnan(ranked)
            rank_sum = np.where(valid_mask, rank_sum + ranked, rank_sum)
            valid_count = np.where(valid_mask, valid_count + 1.0, valid_count)

        # Step 4: average (require >= MIN_VALID_FACTORS valid)
        composite = np.where(
            valid_count >= MIN_VALID_FACTORS,
            rank_sum / valid_count,
            np.nan,
        )

        logger.info(f"{FACTOR_NAME}: nan_ratio={np.isnan(composite).mean():.1%}")

        return FactorData(
            values=composite,
            symbols=common_symbols,
            dates=common_dates,
            name=self.name,
            params=self.params,
        )
