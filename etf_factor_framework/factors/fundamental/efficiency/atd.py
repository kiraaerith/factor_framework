"""
ATD factor (Asset Turnover Dynamics)

Formula: ATD = current_ATO_TTM - previous_ATO_TTM

Where:
  ATO_TTM = revenue_TTM / avg_total_assets
  avg_total_assets = (beginning_TA + ending_TA) / 2
  revenue_TTM = sum of most recent 4 quarters of operating revenue

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_ps_toi_c: total operating income (single quarter, de-cumulated)
  - q_bs_ta_t: total assets

Factor direction: positive (improving asset turnover = better efficiency trend)
Factor category: efficiency

Post-processing:
  1. Mainboard filter (60xxxx, 00xxxx only)
  2. Quantile winsorize: clip to [1st, 99th] percentile per cross-section
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

FACTOR_NAME = "ATD"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _compute_ato_ttm_series(rev_row: np.ndarray, ta_row: np.ndarray, T: int):
    """Compute ATO_TTM time series for a single stock.

    Returns: (T,) array of ATO_TTM values, NaN where not computable.
    Also returns quarter-level ATO values and their start indices for delta.
    """
    valid_rev = np.isfinite(rev_row)
    valid_ta = np.isfinite(ta_row)

    # Detect quarter boundaries for revenue
    rev_quarters = []  # (value, start_idx)
    prev_val = None
    for t in range(T):
        if not valid_rev[t]:
            continue
        if prev_val is None or rev_row[t] != prev_val:
            rev_quarters.append((rev_row[t], t))
            prev_val = rev_row[t]

    # Detect quarter boundaries for total assets
    ta_quarters = []  # (value, start_idx)
    prev_val = None
    for t in range(T):
        if not valid_ta[t]:
            continue
        if prev_val is None or ta_row[t] != prev_val:
            ta_quarters.append((ta_row[t], t))
            prev_val = ta_row[t]

    if len(rev_quarters) < 5 or len(ta_quarters) < 5:
        return np.full(T, np.nan)

    # Compute revenue TTM at each quarter boundary (need 4 quarters)
    rev_ttm_quarters = []  # (ttm_value, start_idx)
    # Loop: ~40 quarters
    for q in range(3, len(rev_quarters)):
        ttm = sum(v for v, _ in rev_quarters[q-3:q+1])
        rev_ttm_quarters.append((ttm, rev_quarters[q][1]))

    # Build a mapping: date_idx -> nearest TA quarter value
    # For avg TA, we need current and 4-quarters-ago TA
    ta_avg_quarters = []  # (avg_ta, start_idx)
    # Loop: ~40 quarters
    for q in range(4, len(ta_quarters)):
        avg = (ta_quarters[q][0] + ta_quarters[q-4][0]) / 2.0
        ta_avg_quarters.append((avg, ta_quarters[q][1]))

    if not rev_ttm_quarters or not ta_avg_quarters:
        return np.full(T, np.nan)

    # Match rev_ttm and ta_avg by closest start_idx
    # Build ATO_TTM at each point where both are available
    ato_quarters = []  # (ato_value, start_idx)

    ta_idx_map = {s: v for v, s in ta_avg_quarters}
    ta_starts = sorted(ta_idx_map.keys())

    for rev_ttm, rev_start in rev_ttm_quarters:
        # Find closest ta_avg at or before rev_start
        best_ta_start = None
        for ts in ta_starts:
            if ts <= rev_start:
                best_ta_start = ts
            else:
                break
        if best_ta_start is None:
            continue
        avg_ta = ta_idx_map[best_ta_start]
        if avg_ta <= 0:
            continue
        ato = rev_ttm / avg_ta
        ato_quarters.append((ato, rev_start))

    if len(ato_quarters) < 2:
        return np.full(T, np.nan)

    # Compute ATD = current ATO - previous ATO, forward fill to daily
    result = np.full(T, np.nan)
    # Loop: ~35 quarters
    for q in range(1, len(ato_quarters)):
        atd = ato_quarters[q][0] - ato_quarters[q-1][0]
        start_t = ato_quarters[q][1]
        end_t = ato_quarters[q+1][1] if q+1 < len(ato_quarters) else T
        result[start_t:end_t] = atd

    return result


class ATD(FundamentalFactorCalculator):
    """
    Asset Turnover Dynamics.
    ATD = current_ATO_TTM - previous_ATO_TTM
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
        rev_v, rev_s, rev_d = fundamental_data.get_daily_panel("q_ps_toi_c")
        ta_v, ta_s, ta_d = fundamental_data.get_daily_panel("q_bs_ta_t")

        if rev_v.size == 0:
            raise ValueError("ATD: get_daily_panel('q_ps_toi_c') returned empty")
        if ta_v.size == 0:
            raise ValueError("ATD: get_daily_panel('q_bs_ta_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(rev_s.tolist()) & set(ta_s.tolist()))
        common_dates_int = sorted(set(rev_d.astype('int64').tolist()) & set(ta_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        rev_si = {s: i for i, s in enumerate(rev_s.tolist())}
        ta_si = {s: i for i, s in enumerate(ta_s.tolist())}
        rev_di = {int(d): i for i, d in enumerate(rev_d.astype('int64'))}
        ta_di = {int(d): i for i, d in enumerate(ta_d.astype('int64'))}

        rev_v = rev_v[np.ix_(
            np.array([rev_si[s] for s in common_syms]),
            np.array([rev_di[d] for d in common_dates_int])
        )]
        ta_v = ta_v[np.ix_(
            np.array([ta_si[s] for s in common_syms]),
            np.array([ta_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        rev_v = rev_v[mainboard_mask].copy()
        ta_v = ta_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        N, T = rev_v.shape

        # Compute ATD per stock
        # Loop: ~N stocks (~3000)
        values = np.full((N, T), np.nan)
        for i in range(N):
            values[i] = _compute_ato_ttm_series(rev_v[i], ta_v[i], T)

        values[~np.isfinite(values)] = np.nan

        # Quantile winsorize per cross-section
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
            warnings.warn(f"ATD NaN ratio is high: {nan_ratio:.1%}")

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
    print("ATD factor smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2022-01-01", end_date="2024-12-31")
    calc = ATD()
    result = calc.calculate(fd)

    print(f"Shape: {result.shape}")
    print(f"NaN ratio: {np.isnan(result.values).mean():.1%}")

    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"Last cross-section: N={len(valid)}, mean={valid.mean():.6f}, "
              f"min={valid.min():.6f}, max={valid.max():.6f}")

    print("\n[Leakage detection]")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31")
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calc, fd_leak)
        status = "LEAK" if report.has_leakage else "OK"
        print(f"  split={sr}: [{status}]")

    print("\n[PASS]")
