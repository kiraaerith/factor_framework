"""
valuation_PB_ROE_Residual factor

PB residual after removing the cross-sectional linear effect of ROE.
For each trading day, run OLS cross-sectionally:
    log(PB_i) = alpha + beta * ROE_i + epsilon_i
The residual epsilon_i is the factor value.

Interpretation: a positive residual means the stock is expensive relative to
its profitability (high PB given ROE); a negative residual means it is cheap.
Factor direction: negative (lower residual = undervalued relative to ROE = better).

Data fields:
  - pb           : lixinger fundamental (daily)
  - q_m_wroe_t   : lixinger financial_statements, weighted ROE TTM (quarterly,
                   forward-filled to daily by FundamentalData)

Notes:
  - Stocks with PB <= 0 are excluded per trading day (log is undefined).
  - Regression is cross-sectional OLS, no winsorization applied to ROE.
  - Only A-share mainboard stocks (60xxxx / 00xxxx) are included in output.
  - Backtest start date 2016-01-01 (lixinger fundamental has full coverage from 2016).
  - q_m_wroe_t has quarterly delay, so leakage detection must run.
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

FACTOR_NAME = "valuation_PB_ROE_Residual"
FACTOR_DIRECTION = -1  # negative: lower residual (cheaper vs ROE) is better


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Cross-sectional OLS: y = alpha + beta * x + residual.
    NaN positions in y or x are excluded from the regression;
    returns NaN for those positions.
    Returns residual array of the same length as y.
    """
    residuals = np.full(len(y), np.nan)
    valid = ~np.isnan(y) & ~np.isnan(x)
    n = valid.sum()
    if n < 3:
        return residuals
    yv = y[valid]
    xv = x[valid]
    A = np.column_stack([np.ones(n), xv])
    beta, _, _, _ = np.linalg.lstsq(A, yv, rcond=None)
    predicted = beta[0] + beta[1] * xv
    residuals[valid] = yv - predicted
    return residuals


class PB_ROE_Residual(FundamentalFactorCalculator):
    """
    PB-ROE Residual factor.

    For each trading day, regress log(PB) on ROE (TTM, weighted) cross-sectionally.
    The residual represents the portion of PB not explained by profitability.

    Data:
        - pb (daily, lixinger fundamental)
        - q_m_wroe_t (quarterly forward-filled, lixinger financial_statements)
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
        Calculate PB-ROE residual factor.

        Steps:
          1. Load PB panel (daily) and ROE TTM panel (quarterly forward-filled).
          2. Intersect symbols, apply mainboard filter.
          3. For each trading day, compute log(PB), then OLS residual on ROE.

        Returns:
            FactorData: N_mainboard x T, values = OLS residual of log(PB) on ROE.
        """
        # --- 1. Load panels ---
        pb_vals, pb_syms, pb_dates = fundamental_data.get_valuation_panel("pb")
        roe_vals, roe_syms, roe_dates = fundamental_data.get_daily_panel("q_m_wroe_t")

        if pb_vals.size == 0:
            raise ValueError(
                "PB_ROE_Residual: get_valuation_panel('pb') returned empty array"
            )
        if roe_vals.size == 0:
            raise ValueError(
                "PB_ROE_Residual: get_daily_panel('q_m_wroe_t') returned empty array"
            )

        # Trading dates must match (both use the same trading calendar)
        # Both panels share the same dates array; verify they match
        if len(pb_dates) != len(roe_dates) or not np.all(pb_dates == roe_dates):
            raise ValueError(
                "PB_ROE_Residual: pb and ROE panels have different date arrays"
            )
        dates = pb_dates

        # --- 2. Intersect symbols, apply mainboard filter ---
        pb_sym_set = {s: i for i, s in enumerate(pb_syms)}
        roe_sym_set = {s: i for i, s in enumerate(roe_syms)}
        common_syms = sorted(
            s for s in pb_sym_set if s in roe_sym_set and _is_mainboard(s)
        )

        if len(common_syms) == 0:
            raise ValueError(
                "PB_ROE_Residual: no common mainboard symbols between PB and ROE panels"
            )

        pb_idx = np.array([pb_sym_set[s] for s in common_syms])
        roe_idx = np.array([roe_sym_set[s] for s in common_syms])

        pb_aligned = pb_vals[pb_idx]    # (N, T)
        roe_aligned = roe_vals[roe_idx]  # (N, T)

        N, T = pb_aligned.shape

        # --- 3. For each trading day, compute OLS residual of log(PB) on ROE ---
        # log(PB): mask PB <= 0 as NaN
        with np.errstate(invalid="ignore", divide="ignore"):
            log_pb = np.where(pb_aligned > 0, np.log(pb_aligned), np.nan)

        residuals = np.full((N, T), np.nan, dtype=np.float64)

        for t in range(T):
            residuals[:, t] = _ols_residuals(log_pb[:, t], roe_aligned[:, t])

        symbols_arr = np.array(common_syms)

        nan_ratio = np.isnan(residuals).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"PB_ROE_Residual NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=residuals,
            symbols=symbols_arr,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python valuation_pb_roe_residual.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("valuation_PB_ROE_Residual smoke test")
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

    print(f"\n[Step 2] Compute PB_ROE_Residual factor")
    calculator = PB_ROE_Residual()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {last5}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"  N valid: {len(valid)}")
        print(f"  mean:   {valid.mean():.4f}")
        print(f"  median: {np.median(valid):.4f}")
        print(f"  std:    {valid.std():.4f}")
        print(f"  min:    {valid.min():.4f}")
        print(f"  max:    {valid.max():.4f}")

    print(f"\n[Step 3] Smoke test assertions")
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 4] Leakage detection (5 split ratios)")
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
