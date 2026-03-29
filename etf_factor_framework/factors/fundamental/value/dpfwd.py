"""
DPFWD factor (Expected Dividend Yield, Forward)

Formula (TTM fallback - FY1 consensus data not yet available):
  DPR = clip(dyr * pe_ttm, 0, 1)          # historical dividend payout ratio
  DPFWD = net_profit_TTM * DPR / mc        # expected forward dividend / market cap
        = min(dyr, 1 / pe_ttm)             # simplified (pe_ttm > 0 path)

When FY1 consensus data becomes available (e.g., Wind API):
  DPFWD = FY1_net_profit * DPR / mc

Data fields:
  - pe_ttm : PE-TTM (market cap / TTM net profit), lixinger.fundamental (daily)
  - dyr    : trailing 12-month dividend yield, lixinger.fundamental (daily)

Factor direction: positive (higher expected dividend yield = cheaper/better)
Factor category: value - forward dividend yield (FY1-based improvement)

Notes:
  - FY1 consensus data not in DB; implementation degrades to TTM fallback.
  - TTM fallback: DPFWD = min(dyr, 1/pe_ttm)
    - For most stocks (DPR <= 1): DPFWD = dyr (same as DP_TTM)
    - For super-dividend stocks (dyr*pe_ttm > 1): DPFWD = 1/pe_ttm (capped at EP)
  - Key difference vs DP_TTM: loss-making stocks (pe_ttm <= 0) yield NaN, not dyr.
  - dyr < 0 is a data error; set to NaN.
  - dyr = 0 (no dividends): DPR = 0, DPFWD = 0. Retained as-is.
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

FACTOR_NAME = "DPFWD"
FACTOR_DIRECTION = 1   # positive: higher expected dividend yield is better
MIN_ABS_PE = 1e-6      # |pe_ttm| below this threshold => near-zero earnings => NaN


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class DPFWD(FundamentalFactorCalculator):
    """
    Expected Dividend Yield (Forward) factor.

    TTM fallback formula (when FY1 consensus is unavailable):
      DPR        = clip(dyr * pe_ttm, 0, 1)
      DPFWD      = net_profit_TTM * DPR / mc = min(dyr, 1/pe_ttm)

    Logic by case:
      pe_ttm > MIN_ABS_PE and dyr >= 0:
        DPFWD = min(dyr, 1/pe_ttm)
      pe_ttm <= 0 (loss-making):
        DPFWD = NaN  (DPR undefined for loss-making stocks)
      dyr < 0 (data error):
        DPFWD = NaN
      pe_ttm is NaN or |pe_ttm| < MIN_ABS_PE:
        DPFWD = NaN

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
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
        Calculate DPFWD daily panel (TTM fallback).

        Returns:
            FactorData: N stocks x T days, values = min(dyr, 1/pe_ttm) (float64)
        """
        pe_values, pe_symbols, pe_dates = fundamental_data.get_valuation_panel("pe_ttm")
        dyr_values, dyr_symbols, dyr_dates = fundamental_data.get_valuation_panel("dyr")

        if pe_values.size == 0:
            raise ValueError("DPFWD: get_valuation_panel('pe_ttm') returned empty array")
        if dyr_values.size == 0:
            raise ValueError("DPFWD: get_valuation_panel('dyr') returned empty array")

        # Apply mainboard filter to each panel before alignment
        pe_mb = np.array([_is_mainboard(s) for s in pe_symbols])
        pe_values  = pe_values[pe_mb]
        pe_symbols = pe_symbols[pe_mb]

        dyr_mb = np.array([_is_mainboard(s) for s in dyr_symbols])
        dyr_values  = dyr_values[dyr_mb]
        dyr_symbols = dyr_symbols[dyr_mb]

        # Align symbols (intersection)
        common_symbols = np.intersect1d(pe_symbols, dyr_symbols)
        if len(common_symbols) == 0:
            raise ValueError("DPFWD: no common symbols between pe_ttm and dyr panels")

        pe_idx  = np.array([np.where(pe_symbols  == s)[0][0] for s in common_symbols])
        dyr_idx = np.array([np.where(dyr_symbols == s)[0][0] for s in common_symbols])

        pe_aligned  = pe_values[pe_idx]    # (N, T)
        dyr_aligned = dyr_values[dyr_idx]  # (N, T)

        # Align dates (intersection)
        common_dates = np.intersect1d(pe_dates, dyr_dates)
        if len(common_dates) == 0:
            raise ValueError("DPFWD: no common dates between pe_ttm and dyr panels")

        pe_date_idx  = np.array([np.where(pe_dates  == d)[0][0] for d in common_dates])
        dyr_date_idx = np.array([np.where(dyr_dates == d)[0][0] for d in common_dates])

        pe_final  = pe_aligned[:,  pe_date_idx]   # (N, T)
        dyr_final = dyr_aligned[:, dyr_date_idx]  # (N, T)

        # Clean dyr: negative values are data errors
        dyr_clean = np.where(dyr_final < 0, np.nan, dyr_final)

        # Compute DPFWD = min(dyr, 1/pe_ttm) for pe_ttm > MIN_ABS_PE
        # Loss-making (pe_ttm <= 0) and near-zero (|pe_ttm| < MIN_ABS_PE) -> NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dpfwd = np.where(
                pe_final > MIN_ABS_PE,
                np.minimum(dyr_clean, 1.0 / pe_final),
                np.nan,
            )

        # Guard against any residual inf (should not occur, but be safe)
        dpfwd = np.where(np.isinf(dpfwd), np.nan, dpfwd)
        dpfwd = dpfwd.astype(np.float64)

        nan_ratio = np.isnan(dpfwd).mean() if dpfwd.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"DPFWD NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=dpfwd,
            symbols=common_symbols,
            dates=common_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python dpfwd.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("DPFWD factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute DPFWD factor")
    calculator = DPFWD()
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
        print(f"  N zeros    : {(valid_cs == 0).sum()} ({(valid_cs == 0).mean():.1%})")
        print(f"  mean       : {valid_cs.mean():.6f}")
        print(f"  std        : {valid_cs.std():.6f}")
        print(f"  min        : {valid_cs.min():.6f}")
        print(f"  max        : {valid_cs.max():.6f}")
        print(f"  median     : {np.median(valid_cs):.6f}")

    # Sanity: check known dividend stocks have reasonable DPFWD
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
