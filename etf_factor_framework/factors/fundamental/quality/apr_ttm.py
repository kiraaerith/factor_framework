"""
APR_TTM factor (Accruals Profit Ratio, TTM)

Formula:
  accrual_profit = operating_profit_TTM - operating_cashflow_TTM
  APR_TTM = accrual_profit / operating_profit_TTM

Higher APR = higher accrual ratio = lower earnings quality.

Data fields (lixinger.financial_statements):
  - q_ps_op_c: operating profit (single quarter) -> manual TTM
  - q_cfs_ncffoa_ttm: net cash flow from operating activities (TTM, pre-computed)

Factor direction: negative (lower APR = better earnings quality)

Post-processing:
  1. Mainboard filter (60xxxx, 00xxxx only)
  2. operating_profit_TTM near zero -> NaN
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

FACTOR_NAME = "APR_TTM"
FACTOR_DIRECTION = -1  # negative: lower accrual ratio = better quality
WINSOR_LO = 1.0
WINSOR_HI = 99.0
MIN_ABS_OP = 1e-6  # minimum |operating_profit| to avoid division instability


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class APR_TTM(FundamentalFactorCalculator):
    """
    Accruals Profit Ratio (TTM).
    APR = (operating_profit_TTM - operating_cashflow_TTM) / operating_profit_TTM
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
        cfo_v, cfo_s, cfo_d = fundamental_data.get_daily_panel("q_cfs_ncffoa_ttm")

        if op_v.size == 0:
            raise ValueError("APR_TTM: get_daily_panel('q_ps_op_c') returned empty")
        if cfo_v.size == 0:
            raise ValueError("APR_TTM: get_daily_panel('q_cfs_ncffoa_ttm') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(op_s.tolist()) & set(cfo_s.tolist()))
        common_dates_int = sorted(set(op_d.astype('int64').tolist()) & set(cfo_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        op_si = {s: i for i, s in enumerate(op_s.tolist())}
        cfo_si = {s: i for i, s in enumerate(cfo_s.tolist())}
        op_di = {int(d): i for i, d in enumerate(op_d.astype('int64'))}
        cfo_di = {int(d): i for i, d in enumerate(cfo_d.astype('int64'))}

        op_v = op_v[np.ix_(
            np.array([op_si[s] for s in common_syms]),
            np.array([op_di[d] for d in common_dates_int])
        )]
        cfo_v = cfo_v[np.ix_(
            np.array([cfo_si[s] for s in common_syms]),
            np.array([cfo_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        op_v = op_v[mainboard_mask].copy()
        cfo_v = cfo_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        # Compute operating profit TTM from single-quarter data
        op_ttm = _compute_ttm(op_v)

        # APR = (op_ttm - cfo_ttm) / op_ttm
        # cfo_v is already TTM
        # Filter: |op_ttm| near zero -> NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            accrual = op_ttm - cfo_v
            values = accrual / op_ttm

        # Set NaN where op_ttm is too small
        values[np.abs(op_ttm) < MIN_ABS_OP] = np.nan
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
            warnings.warn(f"APR_TTM NaN ratio is high: {nan_ratio:.1%}")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )
