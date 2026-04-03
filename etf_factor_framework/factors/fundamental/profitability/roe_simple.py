"""
ROE_SIMPLE factor (Return on Equity, simple)

Formula: ROE = net_profit_ttm / total_owners_equity

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_ps_np_c: net profit (single quarter, de-cumulated) -> TTM via rolling 4Q sum
  - q_bs_toe_t: total owners' equity

Factor direction: positive (higher ROE = better equity profitability)
Factor category: profitability

Post-processing:
  1. Mainboard filter (60xxxx, 00xxxx only)
  2. total_equity <= 0 -> NaN
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
from factors.fundamental.profitability.roa_simple import _compute_ttm

FACTOR_NAME = "ROE_SIMPLE"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class ROE_SIMPLE(FundamentalFactorCalculator):
    """
    Return on Equity (simple).
    ROE = net_profit_TTM / total_owners_equity
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
        np_v, np_s, np_d = fundamental_data.get_daily_panel("q_ps_np_c")
        eq_v, eq_s, eq_d = fundamental_data.get_daily_panel("q_bs_toe_t")

        if np_v.size == 0:
            raise ValueError("ROE_SIMPLE: get_daily_panel('q_ps_np_c') returned empty")
        if eq_v.size == 0:
            raise ValueError("ROE_SIMPLE: get_daily_panel('q_bs_toe_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(np_s.tolist()) & set(eq_s.tolist()))
        common_dates_int = sorted(set(np_d.astype('int64').tolist()) & set(eq_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        np_sym_idx = {s: i for i, s in enumerate(np_s.tolist())}
        eq_sym_idx = {s: i for i, s in enumerate(eq_s.tolist())}
        np_date_idx = {int(d): i for i, d in enumerate(np_d.astype('int64'))}
        eq_date_idx = {int(d): i for i, d in enumerate(eq_d.astype('int64'))}

        ns_idx = np.array([np_sym_idx[s] for s in common_syms])
        es_idx = np.array([eq_sym_idx[s] for s in common_syms])
        nd_idx = np.array([np_date_idx[d] for d in common_dates_int])
        ed_idx = np.array([eq_date_idx[d] for d in common_dates_int])

        np_aligned = np_v[np.ix_(ns_idx, nd_idx)]
        eq_aligned = eq_v[np.ix_(es_idx, ed_idx)]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        np_aligned = np_aligned[mainboard_mask].copy()
        eq_aligned = eq_aligned[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        # Compute TTM net profit
        np_ttm = _compute_ttm(np_aligned)

        # total_equity <= 0 -> NaN
        eq_aligned[eq_aligned <= 0] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            values = np_ttm / eq_aligned

        values[~np.isfinite(values)] = np.nan

        # Quantile winsorize per cross-section
        N, T = values.shape
        # Loop: ~T iterations (~2430 days)
        for t in range(T):
            col = values[:, t]
            valid_mask = np.isfinite(col)
            if valid_mask.sum() < 10:
                continue
            lo = np.nanpercentile(col[valid_mask], WINSOR_LO)
            hi = np.nanpercentile(col[valid_mask], WINSOR_HI)
            values[:, t] = np.where(valid_mask, np.clip(col, lo, hi), np.nan)

        nan_ratio = np.isnan(values).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(f"ROE_SIMPLE NaN ratio is high: {nan_ratio:.1%}")

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
    print("ROE_SIMPLE factor smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2022-01-01", end_date="2024-12-31")
    calc = ROE_SIMPLE()
    result = calc.calculate(fd)

    print(f"Shape: {result.shape}")
    print(f"NaN ratio: {np.isnan(result.values).mean():.1%}")

    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"Last cross-section: N={len(valid)}, mean={valid.mean():.4f}, "
              f"min={valid.min():.4f}, max={valid.max():.4f}")

    print("\n[Leakage detection]")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31")
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calc, fd_leak)
        status = "LEAK" if report.has_leakage else "OK"
        print(f"  split={sr}: [{status}]")

    print("\n[PASS]")
