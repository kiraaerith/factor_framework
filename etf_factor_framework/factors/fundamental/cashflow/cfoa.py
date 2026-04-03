"""
CFOA factor (Cash Flow from Operations to Assets)

Formula: CFOA = operating_cashflow_ttm / total_assets

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_cfs_ncffoa_ttm: net cash flow from operating activities (TTM)
  - q_bs_ta_t: total assets

Factor direction: positive (higher CFOA = better asset cash generation)
Factor category: cashflow

Post-processing:
  1. Mainboard filter (60xxxx, 00xxxx only)
  2. total_assets <= 0 -> NaN
  3. Quantile winsorize: clip to [1st, 99th] percentile per cross-section
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "CFOA"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class CFOA(FundamentalFactorCalculator):
    """
    Cash Flow from Operations to Assets.
    CFOA = q_cfs_ncffoa_ttm / q_bs_ta_t
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
        cfo_v, cfo_s, cfo_d = fundamental_data.get_daily_panel("q_cfs_ncffoa_ttm")
        ta_v, ta_s, ta_d = fundamental_data.get_daily_panel("q_bs_ta_t")

        if cfo_v.size == 0:
            raise ValueError("CFOA: get_daily_panel('q_cfs_ncffoa_ttm') returned empty")
        if ta_v.size == 0:
            raise ValueError("CFOA: get_daily_panel('q_bs_ta_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(cfo_s.tolist()) & set(ta_s.tolist()))
        common_dates_int = sorted(set(cfo_d.astype('int64').tolist()) & set(ta_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        cfo_sym_idx = {s: i for i, s in enumerate(cfo_s.tolist())}
        ta_sym_idx = {s: i for i, s in enumerate(ta_s.tolist())}
        cfo_date_idx = {int(d): i for i, d in enumerate(cfo_d.astype('int64'))}
        ta_date_idx = {int(d): i for i, d in enumerate(ta_d.astype('int64'))}

        cs_idx = np.array([cfo_sym_idx[s] for s in common_syms])
        ts_idx = np.array([ta_sym_idx[s] for s in common_syms])
        cd_idx = np.array([cfo_date_idx[d] for d in common_dates_int])
        td_idx = np.array([ta_date_idx[d] for d in common_dates_int])

        cfo_aligned = cfo_v[np.ix_(cs_idx, cd_idx)]
        ta_aligned = ta_v[np.ix_(ts_idx, td_idx)]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        cfo_aligned = cfo_aligned[mainboard_mask].copy()
        ta_aligned = ta_aligned[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        # Filter: total_assets <= 0 -> NaN
        ta_aligned[ta_aligned <= 0] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            values = cfo_aligned / ta_aligned

        # Replace inf with NaN
        values[~np.isfinite(values)] = np.nan

        # Quantile winsorize per cross-section
        N, T = values.shape
        # Loop: ~T iterations (~2430 days)
        for t in range(T):
            col = values[:, t]
            valid = np.isfinite(col)
            if valid.sum() < 10:
                continue
            lo = np.nanpercentile(col[valid], WINSOR_LO)
            hi = np.nanpercentile(col[valid], WINSOR_HI)
            col_clipped = np.where(valid, np.clip(col, lo, hi), np.nan)
            values[:, t] = col_clipped

        nan_ratio = np.isnan(values).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(f"CFOA NaN ratio is high: {nan_ratio:.1%}")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("CFOA factor smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2022-01-01", end_date="2024-12-31")
    calc = CFOA()
    result = calc.calculate(fd)

    print(f"Shape: {result.shape}")
    print(f"Symbols: {len(result.symbols)}")
    print(f"Date range: {result.dates[0]} ~ {result.dates[-1]}")
    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"Last cross-section: N={len(valid)}, mean={valid.mean():.4f}, "
              f"min={valid.min():.4f}, max={valid.max():.4f}")

    # Leakage detection
    print("\n[Leakage detection]")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31")
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calc, fd_leak)
        status = "LEAK" if report.has_leakage else "OK"
        print(f"  split={sr}: [{status}]")

    print("\n[PASS]")
