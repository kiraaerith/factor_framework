"""
ROA_SIMPLE factor (Return on Assets, simple)

Formula: ROA = net_profit_ttm / total_assets

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_ps_np_c: net profit (single quarter, de-cumulated)
  - q_bs_ta_t: total assets

TTM: rolling sum of most recent 4 quarters of net profit.

Factor direction: positive (higher ROA = better asset profitability)
Factor category: profitability

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

FACTOR_NAME = "ROA_SIMPLE"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _compute_ttm(values: np.ndarray) -> np.ndarray:
    """Compute TTM (trailing 4 quarters) from ffilled daily panel.
    values: (N, T) single-quarter data forward-filled to daily.
    Returns: (N, T) TTM values.
    """
    N, T = values.shape
    ttm = np.full((N, T), np.nan)

    # Loop: ~N stocks (~3000)
    for i in range(N):
        row = values[i]
        valid = np.isfinite(row)
        if valid.sum() < 4:
            continue

        # Detect quarter boundaries (value changes)
        quarter_vals = []
        quarter_starts = []
        prev_val = None
        for t in range(T):
            if not valid[t]:
                continue
            if prev_val is None or row[t] != prev_val:
                quarter_vals.append(row[t])
                quarter_starts.append(t)
                prev_val = row[t]

        # Rolling 4-quarter sum
        # Loop: ~40 quarters per stock
        for q in range(3, len(quarter_vals)):
            ttm_val = sum(quarter_vals[q-3:q+1])
            start_t = quarter_starts[q]
            end_t = quarter_starts[q+1] if q+1 < len(quarter_vals) else T
            ttm[i, start_t:end_t] = ttm_val

    return ttm


class ROA_SIMPLE(FundamentalFactorCalculator):
    """
    Return on Assets (simple).
    ROA = net_profit_TTM / total_assets
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
        ta_v, ta_s, ta_d = fundamental_data.get_daily_panel("q_bs_ta_t")

        if np_v.size == 0:
            raise ValueError("ROA_SIMPLE: get_daily_panel('q_ps_np_c') returned empty")
        if ta_v.size == 0:
            raise ValueError("ROA_SIMPLE: get_daily_panel('q_bs_ta_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(np_s.tolist()) & set(ta_s.tolist()))
        common_dates_int = sorted(set(np_d.astype('int64').tolist()) & set(ta_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        np_si = {s: i for i, s in enumerate(np_s.tolist())}
        ta_si = {s: i for i, s in enumerate(ta_s.tolist())}
        np_di = {int(d): i for i, d in enumerate(np_d.astype('int64'))}
        ta_di = {int(d): i for i, d in enumerate(ta_d.astype('int64'))}

        np_v = np_v[np.ix_(
            np.array([np_si[s] for s in common_syms]),
            np.array([np_di[d] for d in common_dates_int])
        )]
        ta_v = ta_v[np.ix_(
            np.array([ta_si[s] for s in common_syms]),
            np.array([ta_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        np_v = np_v[mainboard_mask].copy()
        ta_v = ta_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        # Compute TTM net profit
        np_ttm = _compute_ttm(np_v)

        # total_assets <= 0 -> NaN
        ta_v[ta_v <= 0] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            values = np_ttm / ta_v

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
            warnings.warn(f"ROA_SIMPLE NaN ratio is high: {nan_ratio:.1%}")

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
    print("ROA_SIMPLE factor smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2022-01-01", end_date="2024-12-31")
    calc = ROA_SIMPLE()
    result = calc.calculate(fd)

    print(f"Shape: {result.shape}")
    print(f"Symbols: {len(result.symbols)}")
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
