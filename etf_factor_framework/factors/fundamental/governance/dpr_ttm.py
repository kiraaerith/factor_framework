"""
DPR_TTM factor (Dividend Payout Ratio, TTM)

Formula: DPR_TTM = cash_dividend_TTM / net_profit_to_parent_TTM

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_cfs_cpfdapdoi_c: cash paid for dividends, profit distribution or interest (single quarter)
  - q_ps_npatoshopc_c: net profit attributable to shareholders of parent company (single quarter)

Both fields are converted to TTM via rolling 4-quarter sum.

Factor direction: positive (higher payout ratio = better governance/quality signal)

Post-processing:
  1. Mainboard filter (60xxxx, 00xxxx only)
  2. net_profit_TTM <= 0 -> NaN (loss-making firms excluded)
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

FACTOR_NAME = "DPR_TTM"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class DPR_TTM(FundamentalFactorCalculator):
    """
    Dividend Payout Ratio (TTM).
    DPR = cash_dividend_TTM / net_profit_to_parent_TTM
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
        div_v, div_s, div_d = fundamental_data.get_daily_panel("q_cfs_cpfdapdoi_c")
        np_v, np_s, np_d = fundamental_data.get_daily_panel("q_ps_npatoshopc_c")

        if div_v.size == 0:
            raise ValueError("DPR_TTM: get_daily_panel('q_cfs_cpfdapdoi_c') returned empty")
        if np_v.size == 0:
            raise ValueError("DPR_TTM: get_daily_panel('q_ps_npatoshopc_c') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(div_s.tolist()) & set(np_s.tolist()))
        common_dates_int = sorted(set(div_d.astype('int64').tolist()) & set(np_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        div_si = {s: i for i, s in enumerate(div_s.tolist())}
        np_si = {s: i for i, s in enumerate(np_s.tolist())}
        div_di = {int(d): i for i, d in enumerate(div_d.astype('int64'))}
        np_di = {int(d): i for i, d in enumerate(np_d.astype('int64'))}

        div_v = div_v[np.ix_(
            np.array([div_si[s] for s in common_syms]),
            np.array([div_di[d] for d in common_dates_int])
        )]
        np_v = np_v[np.ix_(
            np.array([np_si[s] for s in common_syms]),
            np.array([np_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        div_v = div_v[mainboard_mask].copy()
        np_v = np_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        # Compute TTM for both
        div_ttm = _compute_ttm(div_v)
        np_ttm = _compute_ttm(np_v)

        # Only compute for profitable firms (np_ttm > 0)
        np_ttm[np_ttm <= 0] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            values = div_ttm / np_ttm

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
            warnings.warn(f"DPR_TTM NaN ratio is high: {nan_ratio:.1%}")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )
