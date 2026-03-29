"""
SalG_YoY factor (Revenue Year-over-Year Growth Rate)

Data fields:
  - q_ps_toi_c_y2y : lixinger financial_statements
                     (quarterly, single-quarter total operating income YoY growth rate,
                      forward-filled to daily by FundamentalData)

Factor direction: positive (higher revenue growth is better)
Factor category: growth - revenue growth

Notes:
  - q_ps_toi_c_y2y is quarterly data (has reporting delay), so leakage detection must run.
  - The field is already the YoY growth rate (%), no further computation needed.
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

FACTOR_NAME = "SalG_YoY"
FACTOR_DIRECTION = 1  # positive: higher revenue YoY growth is better


class SalG_YoY(FundamentalFactorCalculator):
    """
    Revenue Year-over-Year Growth Rate factor.

    Uses:
      - q_ps_toi_c_y2y: quarterly single-quarter total operating income YoY growth rate
                        from lixinger financial_statements
                        (forward-filled to daily by FundamentalData)
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
        Retrieve q_ps_toi_c_y2y daily panel directly.

        Returns:
            FactorData: N stocks x T days, values = revenue YoY growth rate (%)
        """
        values, symbols, dates = fundamental_data.get_daily_panel("q_ps_toi_c_y2y")

        if values.size == 0:
            raise ValueError(
                "SalG_YoY: get_daily_panel('q_ps_toi_c_y2y') returned empty array"
            )

        nan_ratio = np.isnan(values).sum() / values.size
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"SalG_YoY NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python salg_yoy.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("SalG_YoY factor smoke test")
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

    print(f"\n[Step 2] Compute SalG_YoY factor")
    calculator = SalG_YoY()
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
