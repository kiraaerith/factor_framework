"""
SP_TTM factor (Sales-to-Price, TTM)

Formula: SP_TTM = 1 / ps_ttm

Data fields:
  - ps_ttm : PS-TTM (market cap / TTM operating revenue)
             lixinger.fundamental (daily)

Factor direction: positive (higher SP = cheaper valuation relative to revenue = better)
Factor category: value - classic value

Notes:
  - ps_ttm is daily data, so this is a daily-updated factor with no reporting delay.
  - Stocks with |ps_ttm| < 1e-6 (near-zero revenue) are set to NaN.
  - Stocks with negative ps_ttm (negative revenue, extremely rare) yield SP_TTM < 0, kept as-is.
  - Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only.
"""

import os
import re
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

FACTOR_NAME = "SP_TTM"
FACTOR_DIRECTION = 1   # positive: higher SP (cheaper valuation) is better
MIN_ABS_PS = 1e-6      # |ps_ttm| below this threshold => near-zero revenue => NaN


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class SP_TTM(FundamentalFactorCalculator):
    """
    Sales-to-Price (TTM) factor.

    Computes SP_TTM = 1 / ps_ttm for each stock on each trading day.
    Uses lixinger.fundamental.ps_ttm (daily, already TTM-based).

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
    Stocks with |ps_ttm| < 1e-6 are set to NaN (near-zero operating revenue).
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

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        Calculate SP_TTM daily panel.

        Returns:
            FactorData: N stocks x T days, values = 1/ps_ttm (float64)
        """
        ps_values, symbols, dates = fundamental_data.get_valuation_panel("ps_ttm")

        # Apply mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        ps_values = ps_values[mainboard_mask]
        symbols = symbols[mainboard_mask]

        # Compute SP_TTM = 1 / ps_ttm
        # Suppress division by zero warnings - handled explicitly below
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sp_values = np.where(
                np.abs(ps_values) < MIN_ABS_PS,
                np.nan,
                1.0 / ps_values,
            )

        # Replace any inf that may have slipped through
        sp_values = np.where(np.isinf(sp_values), np.nan, sp_values)
        sp_values = sp_values.astype(np.float64)

        nan_ratio = np.isnan(sp_values).mean() if sp_values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"SP_TTM NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=sp_values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python sp_ttm.py)
# Uses full market with short date range to ensure enough cross-section.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("SP_TTM factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute SP_TTM factor")
    calculator = SP_TTM()
    result = calculator.calculate(fd)

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range   : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # ---- Smoke-test assertions ----------------------------------------
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, \
        f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"NaN ratio    : {nan_ratio:.1%}")

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid    : {len(valid_cs)}")
        print(f"  mean       : {valid_cs.mean():.6f}")
        print(f"  std        : {valid_cs.std():.6f}")
        print(f"  min        : {valid_cs.min():.4f}")
        print(f"  max        : {valid_cs.max():.4f}")
        print(f"  median     : {np.median(valid_cs):.6f}")

    # Sanity: check known stocks have positive SP_TTM
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]
    fd_test = FundamentalData(
        start_date="2024-01-01",
        end_date="2024-12-31",
        stock_codes=TEST_CODES,
    )
    result_test = calculator.calculate(fd_test)
    print(f"\nSample values (5 test stocks, last 5 dates):")
    for i, sym in enumerate(result_test.symbols):
        row = result_test.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {np.round(last5, 4)}")

    print(f"\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31", stock_codes=None)
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
