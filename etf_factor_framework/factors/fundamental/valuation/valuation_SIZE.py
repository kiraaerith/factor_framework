"""
valuation_SIZE factor (Small Cap Size)

Data fields:
  - mc: total market cap (yuan) from lixinger.fundamental (daily)

Factor direction: negative (smaller cap = better expected return)
Factor category: valuation - size

Notes:
  - SIZE = ln(mc), mc > 0 only (mc <= 0 filtered to NaN)
  - Daily data, no reporting delay, no future leakage risk
  - Post-processing: skip market-cap neutralization (factor IS market cap)
    Only industry neutralization is applied, NOT size neutralization.
"""

import os
import sys
import re

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FIELD = "mc"
FACTOR_NAME = "valuation_SIZE"
FACTOR_DIRECTION = -1  # negative: smaller market cap is better


def _is_mainboard(symbol: str) -> bool:
    """Check if stock belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class valuation_SIZE(FundamentalFactorCalculator):
    """
    Small Cap Size factor.

    SIZE = ln(total market cap)

    Uses lixinger fundamental.mc (daily total market cap in yuan).
    Filters mc <= 0 to NaN before log transform.
    Only A-share mainboard stocks are included.

    Post-processing note: market-cap neutralization MUST be skipped for this
    factor (the factor itself IS market cap). Only industry neutralization applies.
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"field": FIELD, "direction": FACTOR_DIRECTION}

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        Compute SIZE = ln(mc) daily panel.

        Returns:
            FactorData: N stocks x T days, values = ln(market cap)
        """
        values, symbols, dates = fundamental_data.get_valuation_panel(FIELD)

        if values.size == 0:
            raise ValueError(
                f"valuation_SIZE: get_valuation_panel('{FIELD}') returned empty array. "
                "Check lixinger database for mc field."
            )

        # Filter to A-share mainboard only
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        values = values[mainboard_mask]
        symbols = symbols[mainboard_mask]

        # Filter mc <= 0 to NaN, then apply log transform
        values = values.copy()
        values[values <= 0] = np.nan
        with np.errstate(invalid='ignore'):
            values = np.log(values)

        nan_ratio = np.isnan(values).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"valuation_SIZE NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python valuation_SIZE.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("valuation_SIZE factor smoke test")
    print("=" * 60)

    TEST_START = "2022-01-01"
    TEST_END   = "2024-12-31"
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print(f"\n[Step 2] Compute valuation_SIZE factor")
    calculator = valuation_SIZE()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ {pd.Timestamp(result.dates[-1]).date()}")

    # Smoke test assertions
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"NaN ratio: {nan_ratio:.1%}")
    print(f"\n[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    print(f"\nSample ln(mc) values (last 5 dates) per stock:")
    target_symbols = [f"SHSE.{c}" if c.startswith('6') else f"SZSE.{c}" for c in TEST_CODES]
    for sym in target_symbols:
        idx = np.where(result.symbols == sym)[0]
        if len(idx):
            row = result.values[idx[0]]
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
