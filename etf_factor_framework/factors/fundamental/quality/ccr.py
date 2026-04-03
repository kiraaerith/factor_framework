"""
CCR factor (Cash to Current Liabilities Ratio)

Formula: CCR = operating_cashflow_TTM / current_liabilities

Data fields (lixinger.financial_statements):
  - q_cfs_ncffoa_ttm: net cash flow from operating activities (TTM)
  - q_bs_cl_t: total current liabilities

Factor direction: positive (higher CCR = stronger short-term solvency)

Post-processing:
  1. Mainboard filter (60xxxx, 00xxxx only)
  2. current_liabilities <= 0 -> NaN
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

FACTOR_NAME = "CCR"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class CCR(FundamentalFactorCalculator):
    """
    Cash to Current Liabilities Ratio.
    CCR = operating_cashflow_TTM / current_liabilities
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
        cl_v, cl_s, cl_d = fundamental_data.get_daily_panel("q_bs_cl_t")

        if cfo_v.size == 0:
            raise ValueError("CCR: get_daily_panel('q_cfs_ncffoa_ttm') returned empty")
        if cl_v.size == 0:
            raise ValueError("CCR: get_daily_panel('q_bs_cl_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(cfo_s.tolist()) & set(cl_s.tolist()))
        common_dates_int = sorted(set(cfo_d.astype('int64').tolist()) & set(cl_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        cfo_si = {s: i for i, s in enumerate(cfo_s.tolist())}
        cl_si = {s: i for i, s in enumerate(cl_s.tolist())}
        cfo_di = {int(d): i for i, d in enumerate(cfo_d.astype('int64'))}
        cl_di = {int(d): i for i, d in enumerate(cl_d.astype('int64'))}

        cfo_v = cfo_v[np.ix_(
            np.array([cfo_si[s] for s in common_syms]),
            np.array([cfo_di[d] for d in common_dates_int])
        )]
        cl_v = cl_v[np.ix_(
            np.array([cl_si[s] for s in common_syms]),
            np.array([cl_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        cfo_v = cfo_v[mainboard_mask].copy()
        cl_v = cl_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        # current_liabilities <= 0 -> NaN
        cl_v[cl_v <= 0] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            values = cfo_v / cl_v

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
            warnings.warn(f"CCR NaN ratio is high: {nan_ratio:.1%}")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )
