"""
valuation_PB_ROE_Residual_V2 factor

V2 improvement over original: industry-within regression instead of whole-market.

For each trading day, within each industry group, run OLS cross-sectionally:
    log(PB_i) = alpha + beta * ROE_i + epsilon_i
The residual epsilon_i is the factor value.

Difference from V1: V1 runs one single regression across all stocks;
V2 groups stocks by industry (lixinger classification) and runs separate
regressions per industry, so residuals reflect within-industry relative
valuation rather than cross-industry effects.

Data fields:
  - pb           : lixinger fundamental (daily)
  - q_m_wroe_t   : lixinger financial_statements, weighted ROE TTM
  - industry     : lixinger company_list (static)
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

FACTOR_NAME = "valuation_PB_ROE_Residual_V2"
FACTOR_DIRECTION = -1


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS: y = alpha + beta * x + residual. Returns residual array."""
    residuals = np.full(len(y), np.nan)
    valid = ~np.isnan(y) & ~np.isnan(x)
    n = valid.sum()
    if n < 8:  # minimum sample size per industry
        return residuals
    yv = y[valid]
    xv = x[valid]
    A = np.column_stack([np.ones(n), xv])
    beta, _, _, _ = np.linalg.lstsq(A, yv, rcond=None)
    predicted = beta[0] + beta[1] * xv
    residuals[valid] = yv - predicted
    return residuals


class PB_ROE_Residual_V2(FundamentalFactorCalculator):
    """
    PB-ROE Residual factor V2: industry-within regression.
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
        # --- 1. Load panels ---
        pb_vals, pb_syms, pb_dates = fundamental_data.get_valuation_panel("pb")
        roe_vals, roe_syms, roe_dates = fundamental_data.get_daily_panel("q_m_wroe_t")

        if pb_vals.size == 0:
            raise ValueError("V2: get_valuation_panel('pb') returned empty")
        if roe_vals.size == 0:
            raise ValueError("V2: get_daily_panel('q_m_wroe_t') returned empty")

        if len(pb_dates) != len(roe_dates) or not np.all(pb_dates == roe_dates):
            raise ValueError("V2: pb and ROE panels have different date arrays")
        dates = pb_dates

        # --- 2. Load industry map ---
        industry_map = fundamental_data.get_industry_map()

        # --- 3. Intersect symbols, apply mainboard filter ---
        pb_sym_set = {s: i for i, s in enumerate(pb_syms)}
        roe_sym_set = {s: i for i, s in enumerate(roe_syms)}
        common_syms = sorted(
            s for s in pb_sym_set
            if s in roe_sym_set and _is_mainboard(s) and s in industry_map
        )

        if len(common_syms) == 0:
            raise ValueError("V2: no common mainboard symbols with industry info")

        pb_idx = np.array([pb_sym_set[s] for s in common_syms])
        roe_idx = np.array([roe_sym_set[s] for s in common_syms])

        pb_aligned = pb_vals[pb_idx]    # (N, T)
        roe_aligned = roe_vals[roe_idx]  # (N, T)

        N, T = pb_aligned.shape

        # Build industry group indices
        sym_industries = [industry_map[s] for s in common_syms]
        unique_industries = sorted(set(sym_industries))
        industry_indices = {}
        for ind in unique_industries:
            industry_indices[ind] = np.array(
                [i for i, si in enumerate(sym_industries) if si == ind]
            )

        # --- 4. For each trading day, compute industry-within OLS residual ---
        with np.errstate(invalid="ignore", divide="ignore"):
            log_pb = np.where(pb_aligned > 0, np.log(pb_aligned), np.nan)

        residuals = np.full((N, T), np.nan, dtype=np.float64)

        for t in range(T):
            for ind, idx in industry_indices.items():
                residuals[idx, t] = _ols_residuals(
                    log_pb[idx, t], roe_aligned[idx, t]
                )

        symbols_arr = np.array(common_syms)

        nan_ratio = np.isnan(residuals).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(f"V2 NaN ratio high: {nan_ratio:.1%}")

        return FactorData(
            values=residuals,
            symbols=symbols_arr,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("valuation_PB_ROE_Residual_V2 smoke test")
    print("=" * 60)

    TEST_START = "2020-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(start_date=TEST_START, end_date=TEST_END, stock_codes=None)

    print(f"\n[Step 2] Compute factor")
    calculator = PB_ROE_Residual_V2()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"\nLast cross-section: N={len(valid)}, mean={valid.mean():.4f}, "
              f"std={valid.std():.4f}")

    assert result.values.ndim == 2
    assert nan_ratio < 0.8
    print(f"\n[PASS] shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # Leakage detection
    print(f"\n[Step 3] Leakage detection")
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
    if leakage_found:
        print("\n[RESULT] LEAKAGE DETECTED")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL PASSED")
