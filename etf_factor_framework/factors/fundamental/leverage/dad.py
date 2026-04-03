"""
DAD factor (Debt-to-Asset Dynamics)

Formula:
  debt_ratio = total_liabilities / total_assets
  DAD = (current_debt_ratio - previous_debt_ratio) / previous_debt_ratio

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_bs_tl_t: total liabilities
  - q_bs_ta_t: total assets

Factor direction: negative (declining leverage = better safety)

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

FACTOR_NAME = "DAD"
FACTOR_DIRECTION = -1  # negative: declining leverage is better
WINSOR_LO = 1.0
WINSOR_HI = 99.0
MIN_ABS_DR = 1e-6  # minimum |debt_ratio| to avoid division instability


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _detect_quarters(row: np.ndarray):
    quarters = []
    prev_val = None
    for t in range(len(row)):
        if not np.isfinite(row[t]):
            continue
        if prev_val is None or row[t] != prev_val:
            quarters.append((row[t], t))
            prev_val = row[t]
    return quarters


def _compute_dad_for_stock(tl_row: np.ndarray, ta_row: np.ndarray, T: int) -> np.ndarray:
    """Compute DAD for a single stock."""
    tl_q = _detect_quarters(tl_row)
    ta_q = _detect_quarters(ta_row)

    if len(tl_q) < 2 or len(ta_q) < 2:
        return np.full(T, np.nan)

    # Match TL and TA quarters by start_idx proximity
    ta_map = {}
    for val, start in ta_q:
        ta_map[start] = val
    ta_starts = sorted(ta_map.keys())

    # Compute debt_ratio at each TL quarter boundary
    dr_quarters = []  # (debt_ratio, start_idx)
    for tl_val, tl_start in tl_q:
        best_ta_start = None
        for ts in ta_starts:
            if ts <= tl_start:
                best_ta_start = ts
            else:
                break
        if best_ta_start is None:
            continue
        ta_val = ta_map[best_ta_start]
        if ta_val <= 0:
            continue
        dr = tl_val / ta_val
        dr_quarters.append((dr, tl_start))

    if len(dr_quarters) < 2:
        return np.full(T, np.nan)

    # DAD = (current_dr - prev_dr) / prev_dr, forward fill
    result = np.full(T, np.nan)
    # Loop: ~40 quarters
    for q in range(1, len(dr_quarters)):
        prev_dr = dr_quarters[q-1][0]
        curr_dr = dr_quarters[q][0]
        if abs(prev_dr) < MIN_ABS_DR:
            continue
        dad = (curr_dr - prev_dr) / prev_dr
        start_t = dr_quarters[q][1]
        end_t = dr_quarters[q+1][1] if q+1 < len(dr_quarters) else T
        result[start_t:end_t] = dad

    return result


class DAD(FundamentalFactorCalculator):
    """
    Debt-to-Asset Dynamics.
    DAD = (current_debt_ratio - previous_debt_ratio) / previous_debt_ratio
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
        tl_v, tl_s, tl_d = fundamental_data.get_daily_panel("q_bs_tl_t")
        ta_v, ta_s, ta_d = fundamental_data.get_daily_panel("q_bs_ta_t")

        if tl_v.size == 0:
            raise ValueError("DAD: get_daily_panel('q_bs_tl_t') returned empty")
        if ta_v.size == 0:
            raise ValueError("DAD: get_daily_panel('q_bs_ta_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(tl_s.tolist()) & set(ta_s.tolist()))
        common_dates_int = sorted(set(tl_d.astype('int64').tolist()) & set(ta_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        tl_si = {s: i for i, s in enumerate(tl_s.tolist())}
        ta_si = {s: i for i, s in enumerate(ta_s.tolist())}
        tl_di = {int(d): i for i, d in enumerate(tl_d.astype('int64'))}
        ta_di = {int(d): i for i, d in enumerate(ta_d.astype('int64'))}

        tl_v = tl_v[np.ix_(
            np.array([tl_si[s] for s in common_syms]),
            np.array([tl_di[d] for d in common_dates_int])
        )]
        ta_v = ta_v[np.ix_(
            np.array([ta_si[s] for s in common_syms]),
            np.array([ta_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        tl_v = tl_v[mainboard_mask].copy()
        ta_v = ta_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        N, T = tl_v.shape

        # Compute DAD per stock
        # Loop: ~N stocks (~3000)
        values = np.full((N, T), np.nan)
        for i in range(N):
            values[i] = _compute_dad_for_stock(tl_v[i], ta_v[i], T)

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
            warnings.warn(f"DAD NaN ratio is high: {nan_ratio:.1%}")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )
