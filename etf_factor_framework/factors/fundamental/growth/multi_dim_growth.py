"""
MULTI_DIM_GROWTH factor (Multi-Dimensional Composite Growth)

Second-order composite factor that synthesizes 8 COMP sub-factors into
6 economic dimensions, then equal-weights across dimensions.

Sub-factors (all pre-computed, rank-normalized to [0, 1]):
  - NI_GROWTH_COMP  : attributable net profit composite growth
  - REV_GROWTH_COMP : revenue composite growth
  - ROE_GROWTH_COMP : ROE composite growth
  - GPM_GROWTH_COMP : gross profit margin composite growth
  - OCF_GROWTH_COMP : operating cash flow composite growth
  - TA_GROWTH_COMP  : total assets composite growth
  - NA_GROWTH_COMP  : net assets composite growth
  - ATO_GROWTH_COMP : asset turnover composite growth

Dimension composition:
  dim1 (profit)      = equal_weight(NI_GROWTH_COMP, REV_GROWTH_COMP)
  dim2 (quality)     = ROE_GROWTH_COMP
  dim3 (pricing)     = GPM_GROWTH_COMP
  dim4 (cashflow)    = OCF_GROWTH_COMP
  dim5 (capital)     = equal_weight(TA_GROWTH_COMP, NA_GROWTH_COMP)
  dim6 (efficiency)  = ATO_GROWTH_COMP

Final synthesis:
  MULTI_DIM_GROWTH = equal_weight(dim1..dim6), min 4 valid dimensions else NaN

Factor direction: positive (higher multi-dimensional growth is better)
Factor category: growth - multi-dimensional composite
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

# Import all 8 COMP sub-factor calculators
from factors.fundamental.growth.ni_growth_comp import NI_GROWTH_COMP
from factors.fundamental.growth.rev_growth_comp import REV_GROWTH_COMP
from factors.fundamental.growth.roe_growth_comp import ROE_GROWTH_COMP
from factors.fundamental.growth.gpm_growth_comp import GPM_GROWTH_COMP
from factors.fundamental.growth.ocf_growth_comp import OCF_GROWTH_COMP
from factors.fundamental.growth.ta_growth_comp import TA_GROWTH_COMP
from factors.fundamental.growth.na_growth_comp import NA_GROWTH_COMP
from factors.fundamental.growth.ato_growth_comp import ATO_GROWTH_COMP

FACTOR_NAME = "MULTI_DIM_GROWTH"
FACTOR_DIRECTION = 1  # positive: higher multi-dimensional growth is better

# Minimum number of valid dimensions required (out of 6) to compute composite
MIN_VALID_DIMS = 4


def _factor_to_df(factor_data: FactorData) -> pd.DataFrame:
    """Convert FactorData to DataFrame with symbol index and date columns."""
    return pd.DataFrame(
        factor_data.values,
        index=factor_data.symbols,
        columns=pd.DatetimeIndex(factor_data.dates),
    )


def _equal_weight_avg_dfs(dfs: list) -> pd.DataFrame:
    """
    Equal-weight nanmean of a list of DataFrames (aligned on a common index/columns).

    Each DataFrame should already be reindexed to the same (symbols, dates) grid.
    Returns a DataFrame; NaN where all inputs are NaN.
    """
    if len(dfs) == 1:
        return dfs[0].copy()
    stacked = np.stack([df.values for df in dfs], axis=0)  # (k, N, T)
    result = np.nanmean(stacked, axis=0)                    # (N, T)
    all_nan = np.all(np.isnan(stacked), axis=0)
    result[all_nan] = np.nan
    return pd.DataFrame(result, index=dfs[0].index, columns=dfs[0].columns)


class MULTI_DIM_GROWTH(FundamentalFactorCalculator):
    """
    Multi-Dimensional Composite Growth factor.

    Synthesizes 8 COMP sub-factors into 6 economic growth dimensions,
    then equal-weights across dimensions (minimum 4 valid required).

    The 8 COMP sub-factors are pre-rank-normalized to [0, 1].
    No additional post-processing is applied to the composite output.
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
            "direction": FACTOR_DIRECTION,
            "dimensions": 6,
            "min_valid_dims": MIN_VALID_DIMS,
            "sub_factors": [
                "NI_GROWTH_COMP", "REV_GROWTH_COMP", "ROE_GROWTH_COMP",
                "GPM_GROWTH_COMP", "OCF_GROWTH_COMP",
                "TA_GROWTH_COMP", "NA_GROWTH_COMP", "ATO_GROWTH_COMP",
            ],
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute MULTI_DIM_GROWTH daily panel.

        Steps:
          1. Compute each of the 8 COMP sub-factors using the same FundamentalData.
          2. Convert each FactorData to DataFrame and align to union (symbol, date) grid.
          3. Compute 6 dimension values via equal-weight nanmean.
          4. Combine 6 dimensions via equal-weight nanmean (min MIN_VALID_DIMS valid).
          5. Return FactorData.

        Returns:
            FactorData: N stocks x T days, values from equal-weight dimension synthesis
        """
        # Step 1: Compute all 8 sub-factors
        sub_calculators = [
            ("NI",  NI_GROWTH_COMP()),
            ("REV", REV_GROWTH_COMP()),
            ("ROE", ROE_GROWTH_COMP()),
            ("GPM", GPM_GROWTH_COMP()),
            ("OCF", OCF_GROWTH_COMP()),
            ("TA",  TA_GROWTH_COMP()),
            ("NA",  NA_GROWTH_COMP()),
            ("ATO", ATO_GROWTH_COMP()),
        ]

        sub_dfs = {}
        for label, calc in sub_calculators:
            fd_result = calc.calculate(fundamental_data)
            sub_dfs[label] = _factor_to_df(fd_result)

        # Step 2: Build union symbol and date grids
        all_symbols = sorted(
            set().union(*[set(df.index.tolist()) for df in sub_dfs.values()])
        )
        all_dates = sorted(
            set().union(*[set(df.columns.tolist()) for df in sub_dfs.values()])
        )
        all_dates = pd.DatetimeIndex(all_dates)

        # Reindex each sub-factor DataFrame to the union grid
        aligned = {}
        for label, df in sub_dfs.items():
            aligned[label] = df.reindex(index=all_symbols, columns=all_dates)

        # Step 3: Compute 6 dimension values
        dim1 = _equal_weight_avg_dfs([aligned["NI"],  aligned["REV"]])   # profit
        dim2 = aligned["ROE"].copy()                                       # quality
        dim3 = aligned["GPM"].copy()                                       # pricing
        dim4 = aligned["OCF"].copy()                                       # cashflow
        dim5 = _equal_weight_avg_dfs([aligned["TA"],  aligned["NA"]])     # capital
        dim6 = aligned["ATO"].copy()                                       # efficiency

        dims = [dim1, dim2, dim3, dim4, dim5, dim6]

        # Step 4: 6-dimension equal-weight nanmean, min MIN_VALID_DIMS valid
        stacked_dims = np.stack([d.values for d in dims], axis=0)  # (6, N, T)
        valid_count = np.sum(~np.isnan(stacked_dims), axis=0)       # (N, T)
        composite = np.nanmean(stacked_dims, axis=0)                 # (N, T)
        composite[valid_count < MIN_VALID_DIMS] = np.nan

        # Validate output
        if composite.size == 0:
            raise ValueError(f"{FACTOR_NAME}: composite output is empty")

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"{FACTOR_NAME} NaN ratio is high: {nan_ratio:.1%}, please check sub-factors"
            )

        symbols_arr = np.array(all_symbols)
        dates_arr = np.array(all_dates, dtype="datetime64[ns]")
        values = composite.astype(np.float64)

        return FactorData(
            values=values,
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python multi_dim_growth.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("MULTI_DIM_GROWTH factor smoke test")
    print("=" * 60)

    TEST_START = "2020-01-01"
    TEST_END   = "2024-12-31"
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print(f"\n[Step 2] Compute MULTI_DIM_GROWTH factor")
    calculator = MULTI_DIM_GROWTH()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ {pd.Timestamp(result.dates[-1]).date()}")

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    # --- Section 4.1 Smoke Test Assertions ---
    print("\n[Assertions]")

    assert result.values.ndim == 2, "values must be 2-D"
    print(f"  [PASS] shape is 2-D: {result.values.shape}")

    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"
    print(f"  [PASS] dtype = float64")

    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"  [PASS] NaN ratio < 80%: {nan_ratio:.1%}")

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"
    print(f"  [PASS] No inf values")

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"
    print(f"  [PASS] Idempotent")

    print(f"\n[PASS] All smoke test assertions passed!")
    print(f"[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {last5}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid: {len(valid_cs)}")
        print(f"  mean: {valid_cs.mean():.4f}")
        print(f"  median: {np.median(valid_cs):.4f}")
        print(f"  min: {valid_cs.min():.4f}")
        print(f"  max: {valid_cs.max():.4f}")
    else:
        print("  No valid values in last cross-section")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2013-01-01", end_date="2025-12-31", stock_codes=None)
    leakage_found = False
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n--- split_ratio={sr} ---")
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calculator, fd_leak)
        report.print_report()
        if report.has_leakage:
            leakage_found = True
            print(f"[FAIL] Leakage detected at split_ratio={sr}")
        else:
            print(f"[OK] No leakage at split_ratio={sr}")

    if leakage_found:
        print("\n[RESULT] LEAKAGE DETECTED")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL PASSED - No leakage")
