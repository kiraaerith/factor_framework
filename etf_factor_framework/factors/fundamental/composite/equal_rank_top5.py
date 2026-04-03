"""
EQUAL_RANK_TOP5: Equal-weight rank composite of top 5 fundamental factors.

Sub-factors (direction):
  1. DYR          (+1) dividend yield
  2. valuation_SIZE (-1) small cap (ln market cap, smaller is better)
  3. NI_ST_FT     (+1) analyst short-term NI forecast revision
  4. RP_EP        (+1) residual earnings-to-price
  5. NA_GROWTH_COMP (+1) net asset growth composite

Method:
  - For each sub-factor, compute cross-sectional percentile rank (0~1)
  - Flip rank for negative-direction factors: rank = 1 - rank
  - Composite = simple average of 5 ranks
  - Higher composite = better (direction = positive)

Factor direction: positive (higher composite rank = better)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "EQUAL_RANK_TOP5"
FACTOR_DIRECTION = 1  # positive: higher composite = better

# Sub-factor configs: (module_path, class_name, direction)
SUB_FACTORS = [
    ("factors.fundamental.value.dyr", "DYR", 1),
    ("factors.fundamental.valuation.valuation_SIZE", "valuation_SIZE", -1),
    ("factors.fundamental.growth.ni_st_ft", "NI_ST_FT", 1),
    ("factors.fundamental.value.rp_ep", "RP_EP", 1),
    ("factors.fundamental.growth.na_growth_comp", "NA_GROWTH_COMP", 1),
]


def _cross_sectional_rank(arr: np.ndarray) -> np.ndarray:
    """
    Compute cross-sectional percentile rank for each column (date).
    NaN stays NaN. Result in [0, 1].

    arr: (N, T) ndarray
    returns: (N, T) ndarray of percentile ranks
    """
    result = np.full_like(arr, np.nan)
    N, T = arr.shape
    # Loop: ~T iterations (~2430 trading days), acceptable
    for t in range(T):
        col = arr[:, t]
        valid = ~np.isnan(col)
        n_valid = valid.sum()
        if n_valid < 2:
            continue
        # argsort of argsort gives rank (0-based)
        order = np.argsort(col[valid])
        ranks = np.empty(order.shape[0], dtype=np.float64)
        ranks[order] = np.arange(order.shape[0], dtype=np.float64)
        # Normalize to [0, 1]
        ranks /= (n_valid - 1)
        result[valid, t] = ranks
    return result


class EQUAL_RANK_TOP5(FundamentalFactorCalculator):
    """
    Equal-weight rank composite of 5 top fundamental factors.
    """

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
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        1. Compute each sub-factor
        2. Align to common symbols/dates
        3. Cross-sectional rank each, flip if negative direction
        4. Average ranks
        """
        import importlib

        # Step 1: compute all sub-factors
        sub_results = []
        for mod_path, cls_name, direction in SUB_FACTORS:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            calculator = cls()
            fd = calculator.calculate(fundamental_data)
            sub_results.append((fd, direction))

        # Step 2: find common symbols and dates
        # Use Python str for symbols, int64 view for dates to avoid type mismatch
        common_symbols_set = None
        common_dates_set = None
        for fd, _ in sub_results:
            syms = set(fd.symbols.tolist())
            dts = set(fd.dates.astype('int64').tolist())
            common_symbols_set = syms if common_symbols_set is None else common_symbols_set & syms
            common_dates_set = dts if common_dates_set is None else common_dates_set & dts

        common_symbols = np.array(sorted(common_symbols_set))
        common_dates = np.array(sorted(common_dates_set), dtype='datetime64[ns]')
        N = len(common_symbols)
        T = len(common_dates)

        if N == 0 or T == 0:
            raise ValueError(
                f"EQUAL_RANK_TOP5: no common symbols/dates. "
                f"symbols={N}, dates={T}"
            )

        # Build symbol/date index maps for each sub-factor
        # Loop: ~5 sub-factors x N symbols, fine
        rank_sum = np.zeros((N, T), dtype=np.float64)
        valid_count = np.zeros((N, T), dtype=np.float64)

        for fd, direction in sub_results:
            # Build index maps via dict lookup (O(1) per symbol)
            sym_to_idx = {s: i for i, s in enumerate(fd.symbols.tolist())}
            date_to_idx = {int(d): i for i, d in enumerate(fd.dates.astype('int64'))}

            # Extract aligned sub-matrix
            sym_indices = np.array([sym_to_idx[s] for s in common_symbols.tolist()])
            date_indices = np.array([date_to_idx[int(d)] for d in common_dates.astype('int64')])
            aligned = fd.values[np.ix_(sym_indices, date_indices)]

            # Cross-sectional rank
            ranked = _cross_sectional_rank(aligned)

            # Flip for negative direction factors (smaller raw = higher rank)
            if direction == -1:
                ranked = np.where(np.isnan(ranked), np.nan, 1.0 - ranked)

            # Accumulate
            valid_mask = ~np.isnan(ranked)
            rank_sum = np.where(valid_mask, rank_sum + ranked, rank_sum)
            valid_count = np.where(valid_mask, valid_count + 1.0, valid_count)

        # Average rank (require at least 3 out of 5 sub-factors valid)
        min_valid = 3
        composite = np.where(
            valid_count >= min_valid,
            rank_sum / valid_count,
            np.nan,
        )

        return FactorData(
            values=composite,
            symbols=common_symbols,
            dates=common_dates,
            name=self.name,
            params=self.params,
        )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("EQUAL_RANK_TOP5 composite factor smoke test")
    print("=" * 60)

    TEST_START = "2022-01-01"
    TEST_END = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(start_date=TEST_START, end_date=TEST_END)

    print(f"\n[Step 2] Compute EQUAL_RANK_TOP5")
    calc = EQUAL_RANK_TOP5()
    result = calc.calculate(fd)

    print(f"\nFactor shape: {result.shape}")
    print(f"Symbols count: {len(result.symbols)}")
    print(f"Date range: {result.dates[0]} ~ {result.dates[-1]}")

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"\nLast cross-section: N={len(valid)}, "
              f"mean={valid.mean():.4f}, min={valid.min():.4f}, max={valid.max():.4f}")

    print("\n[PASS] Smoke test completed")
