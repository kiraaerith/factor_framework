"""
FCF_MC factor (Free Cash Flow Yield = FCF_TTM / Market Cap)

Data fields:
  - q_m_fcf_ttm : lixinger financial_statements (quarterly, forward-filled to daily)
  - mc          : lixinger fundamental (daily market cap, unit: 100M CNY)

Factor direction: positive (higher FCF yield is better)
Factor category: value - cash flow valuation

Notes:
  - FCF is quarterly data (has reporting delay), so leakage detection must run.
  - mc unit is 100M CNY (yi yuan); q_m_fcf_ttm unit is also 100M CNY (yi yuan),
    so the ratio FCF/MC is dimensionless.
  - Symbols in get_daily_panel and get_valuation_panel may differ slightly;
    we intersect them before computing the ratio.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "FCF_MC"
FACTOR_DIRECTION = 1  # positive: higher FCF yield is better


class FCF_MC(FundamentalFactorCalculator):
    """
    Free Cash Flow Yield factor: FCF_TTM / Market Cap

    Uses:
      - q_m_fcf_ttm: quarterly FCF TTM from lixinger financial_statements
                     (forward-filled to daily by FundamentalData)
      - mc:          daily total market cap from lixinger fundamental
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"direction": FACTOR_DIRECTION}

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute FCF_TTM / MC daily panel.

        Returns:
            FactorData: N stocks x T days, values = FCF_TTM / MC (dimensionless)
        """
        # --- load FCF TTM (quarterly, forward-filled to daily) ---
        fcf_values, fcf_symbols, fcf_dates = fundamental_data.get_daily_panel("q_m_fcf_ttm")

        # --- load daily market cap ---
        mc_values, mc_symbols, mc_dates = fundamental_data.get_valuation_panel("mc")

        if fcf_values.size == 0:
            raise ValueError("FCF_MC: get_daily_panel('q_m_fcf_ttm') returned empty array")
        if mc_values.size == 0:
            raise ValueError("FCF_MC: get_valuation_panel('mc') returned empty array")

        # --- align symbols (intersection) ---
        common_symbols = np.intersect1d(fcf_symbols, mc_symbols)
        if len(common_symbols) == 0:
            raise ValueError("FCF_MC: no common symbols between fcf and mc panels")

        fcf_idx = np.array([np.where(fcf_symbols == s)[0][0] for s in common_symbols])
        mc_idx  = np.array([np.where(mc_symbols  == s)[0][0] for s in common_symbols])

        fcf_aligned = fcf_values[fcf_idx]  # (N, T_fcf)
        mc_aligned  = mc_values[mc_idx]    # (N, T_mc)

        # --- align dates (intersection) ---
        # Both panels should have the same trading dates from FundamentalData,
        # but intersect just in case.
        common_dates = np.intersect1d(fcf_dates, mc_dates)
        if len(common_dates) == 0:
            raise ValueError("FCF_MC: no common dates between fcf and mc panels")

        fcf_date_idx = np.array([np.where(fcf_dates == d)[0][0] for d in common_dates])
        mc_date_idx  = np.array([np.where(mc_dates  == d)[0][0] for d in common_dates])

        fcf_final = fcf_aligned[:, fcf_date_idx]  # (N, T)
        mc_final  = mc_aligned[:, mc_date_idx]    # (N, T)

        # --- compute ratio: FCF_TTM / MC ---
        # Set to NaN where MC <= 0 (negative/zero market cap is invalid)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mc_final > 0, fcf_final / mc_final, np.nan)

        nan_ratio = np.isnan(ratio).sum() / ratio.size
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(f"FCF_MC NaN ratio is high: {nan_ratio:.1%}, please check data")

        return FactorData(
            values=ratio,
            symbols=common_symbols,
            dates=common_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python fcf_mc.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("FCF_MC factor smoke test")
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

    print(f"\n[Step 2] Compute FCF_MC factor")
    calculator = FCF_MC()
    factor = calculator.calculate(fd)

    print(f"\nFactor shape: {factor.shape}")
    print(f"Symbols: {factor.symbols}")
    print(f"Date range: {pd.Timestamp(factor.dates[0]).date()} ~ {pd.Timestamp(factor.dates[-1]).date()}")

    nan_ratio = np.isnan(factor.values).sum() / factor.values.size
    print(f"NaN ratio: {nan_ratio:.1%}")

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(factor.symbols):
        row = factor.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {last5}")

    print(f"\nLast cross-section stats:")
    last_cs = factor.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"  N valid: {len(valid)}")
        print(f"  mean: {valid.mean():.4f}")
        print(f"  median: {np.median(valid):.4f}")
        print(f"  min: {valid.min():.4f}")
        print(f"  max: {valid.max():.4f}")

    print(f"\n[Step 3] Leakage detection")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    detector = FundamentalLeakageDetector(split_ratio=0.7)
    report = detector.detect(calculator, fd)
    report.print_report()

    if report.has_leakage:
        print("\n[FAIL] Leakage detected! Check calculate() logic.")
    else:
        print("\n[PASS] No leakage. Factor code is correct.")
