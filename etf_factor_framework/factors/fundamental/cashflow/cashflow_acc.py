"""
cashflow_ACC factor (Cash Flow Accruals)

Formula: ACC = (NI - CFO) / TA
  - NI  : q_ps_np_c        -- single-quarter net profit (lixinger financial_statements)
  - CFO : q_cfs_ncffoa_c   -- single-quarter net operating cash flow (lixinger financial_statements)
  - TA  : q_bs_ta_t        -- total assets, balance sheet (lixinger financial_statements)

High accruals (ACC > 0, or large positive) means earnings are driven by accruals rather
than cash, indicating lower earnings quality and predicting weaker future returns.

Factor direction: negative (higher ACC -> worse future return)
Factor category: cashflow - earnings quality / accruals
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "cashflow_ACC"
FACTOR_DIRECTION = -1  # negative: higher accruals -> worse future return


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class cashflow_ACC(FundamentalFactorCalculator):
    """
    Cash Flow Accruals factor.

    ACC = (NI - CFO) / TA

    Uses single-quarter values from lixinger financial_statements,
    forward-filled to daily by FundamentalData.get_daily_panel().

    Only A-share mainboard stocks (60xxxx, 00xxxx) are included.
    TA <= 0 is set to NaN.
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
        Compute cashflow_ACC daily panel (N stocks x T days).

        Returns:
            FactorData with values = ACC = (NI - CFO) / TA  (dimensionless)
        """
        # --- 1. Load daily panels from financial_statements ---
        ni_v, ni_s, ni_d = fundamental_data.get_daily_panel("q_ps_np_c")
        cfo_v, cfo_s, cfo_d = fundamental_data.get_daily_panel("q_cfs_ncffoa_c")
        ta_v, ta_s, ta_d = fundamental_data.get_daily_panel("q_bs_ta_t")

        if ni_v.size == 0:
            raise ValueError("cashflow_ACC: get_daily_panel('q_ps_np_c') returned empty")
        if cfo_v.size == 0:
            raise ValueError("cashflow_ACC: get_daily_panel('q_cfs_ncffoa_c') returned empty")
        if ta_v.size == 0:
            raise ValueError("cashflow_ACC: get_daily_panel('q_bs_ta_t') returned empty")

        # --- 2. Common symbols: intersection of all three panels ---
        common_syms = np.intersect1d(np.intersect1d(ni_s, cfo_s), ta_s)

        # Apply mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in common_syms])
        common_syms = common_syms[mainboard_mask]

        if len(common_syms) == 0:
            raise ValueError("cashflow_ACC: no mainboard symbols found")

        # --- 3. Use NI dates as the reference trading calendar ---
        common_dates = ni_d

        N = len(common_syms)
        T = len(common_dates)

        def _align(vals, syms, dates, target_syms, target_dates):
            """Align (vals, syms, dates) to (target_syms, target_dates) via index lookup."""
            si = np.searchsorted(syms, target_syms)
            di = np.searchsorted(dates, target_dates)
            # Clip to valid range (symbols and dates must exist)
            si = np.clip(si, 0, len(syms) - 1)
            di = np.clip(di, 0, len(dates) - 1)
            return vals[si][:, di]

        ni_aligned  = _align(ni_v,  ni_s,  ni_d,  common_syms, common_dates)
        cfo_aligned = _align(cfo_v, cfo_s, cfo_d, common_syms, common_dates)
        ta_aligned  = _align(ta_v,  ta_s,  ta_d,  common_syms, common_dates)

        # --- 4. Compute ACC = (NI - CFO) / TA ---
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(
                (ta_aligned > 0)
                & ~np.isnan(ni_aligned)
                & ~np.isnan(cfo_aligned)
                & ~np.isnan(ta_aligned),
                (ni_aligned - cfo_aligned) / ta_aligned,
                np.nan,
            )
        acc = acc.astype(np.float64)

        nan_ratio = np.isnan(acc).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"cashflow_ACC NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=acc,
            symbols=common_syms,
            dates=common_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python cashflow_acc.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("cashflow_ACC factor smoke test")
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

    print(f"\n[Step 2] Validate required fields")
    for field in ["q_ps_np_c", "q_cfs_ncffoa_c", "q_bs_ta_t"]:
        v, s, d = fd.get_daily_panel(field)
        nr = np.isnan(v).mean()
        print(f"  {field}: shape={v.shape}, NaN={nr:.1%}")
        assert v.size > 0, f"{field} returned empty"
        assert nr < 0.8, f"{field} NaN rate too high: {nr:.1%}"
    print("  [PASS] data validation")

    print(f"\n[Step 3] Compute cashflow_ACC factor")
    calculator = cashflow_ACC()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # --- 4.1 Smoke test assertions ---
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\n[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

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
        print(f"  mean:    {valid_cs.mean():.6f}")
        print(f"  median:  {np.median(valid_cs):.6f}")
        print(f"  min:     {valid_cs.min():.6f}")
        print(f"  max:     {valid_cs.max():.6f}")

    print(f"\n[NOTE] Leakage detection is handled in the next step.")
