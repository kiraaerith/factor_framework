"""
OPMD factor (Operating Profit Margin Dynamics)

Formula: OPMD = current_OPM_TTM - previous_OPM_TTM

Where:
  OPM_TTM = operating_profit_TTM / revenue_TTM
  operating_profit_TTM = sum of most recent 4 quarters of q_ps_op_c
  revenue_TTM = sum of most recent 4 quarters of q_ps_toi_c

Factor direction: positive (improving operating margin = better profitability trend)
Factor category: efficiency / profitability

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

FACTOR_NAME = "OPMD"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


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


def _compute_ttm_quarters(quarters):
    result = []
    # Loop: ~40 quarters
    for q in range(3, len(quarters)):
        ttm = sum(v for v, _ in quarters[q-3:q+1])
        result.append((ttm, quarters[q][1]))
    return result


def _compute_opmd_for_stock(op_row: np.ndarray, rev_row: np.ndarray, T: int) -> np.ndarray:
    op_q = _detect_quarters(op_row)
    rev_q = _detect_quarters(rev_row)

    if len(op_q) < 5 or len(rev_q) < 5:
        return np.full(T, np.nan)

    op_ttm_q = _compute_ttm_quarters(op_q)
    rev_ttm_q = _compute_ttm_quarters(rev_q)

    if len(op_ttm_q) < 2 or len(rev_ttm_q) < 2:
        return np.full(T, np.nan)

    rev_map = {s: v for v, s in rev_ttm_q}

    opm_quarters = []
    for op_ttm, start in op_ttm_q:
        best_rev_start = None
        for rs in sorted(rev_map.keys()):
            if rs <= start:
                best_rev_start = rs
            else:
                break
        if best_rev_start is None:
            continue
        rev_ttm = rev_map[best_rev_start]
        if rev_ttm <= 0:
            continue
        opm = op_ttm / rev_ttm
        opm_quarters.append((opm, start))

    if len(opm_quarters) < 2:
        return np.full(T, np.nan)

    result = np.full(T, np.nan)
    # Loop: ~35 quarters
    for q in range(1, len(opm_quarters)):
        opmd = opm_quarters[q][0] - opm_quarters[q-1][0]
        start_t = opm_quarters[q][1]
        end_t = opm_quarters[q+1][1] if q+1 < len(opm_quarters) else T
        result[start_t:end_t] = opmd

    return result


class OPMD(FundamentalFactorCalculator):
    """
    Operating Profit Margin Dynamics.
    OPMD = current_OPM_TTM - previous_OPM_TTM
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
        op_v, op_s, op_d = fundamental_data.get_daily_panel("q_ps_op_c")
        rev_v, rev_s, rev_d = fundamental_data.get_daily_panel("q_ps_toi_c")

        if op_v.size == 0:
            raise ValueError("OPMD: get_daily_panel('q_ps_op_c') returned empty")
        if rev_v.size == 0:
            raise ValueError("OPMD: get_daily_panel('q_ps_toi_c') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(op_s.tolist()) & set(rev_s.tolist()))
        common_dates_int = sorted(set(op_d.astype('int64').tolist()) & set(rev_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        op_si = {s: i for i, s in enumerate(op_s.tolist())}
        rev_si = {s: i for i, s in enumerate(rev_s.tolist())}
        op_di = {int(d): i for i, d in enumerate(op_d.astype('int64'))}
        rev_di = {int(d): i for i, d in enumerate(rev_d.astype('int64'))}

        op_v = op_v[np.ix_(
            np.array([op_si[s] for s in common_syms]),
            np.array([op_di[d] for d in common_dates_int])
        )]
        rev_v = rev_v[np.ix_(
            np.array([rev_si[s] for s in common_syms]),
            np.array([rev_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        op_v = op_v[mainboard_mask].copy()
        rev_v = rev_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        N, T = op_v.shape

        # Compute OPMD per stock
        # Loop: ~N stocks (~3000)
        values = np.full((N, T), np.nan)
        for i in range(N):
            values[i] = _compute_opmd_for_stock(op_v[i], rev_v[i], T)

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
            warnings.warn(f"OPMD NaN ratio is high: {nan_ratio:.1%}")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )
