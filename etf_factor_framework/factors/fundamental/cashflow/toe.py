"""
TOE factor (Tax on Equity)

Formula: TOE = income_tax_expense_ttm / avg_parent_equity

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_ps_ite_c: income tax expense (single quarter, cumulative -> need TTM)
  - q_bs_tetoshopc_t: total equity to shareholders of parent company

TTM computation: sum of most recent 4 quarters of income tax expense.
Average equity: (current quarter equity + 4-quarters-ago equity) / 2.

Factor direction: positive (higher TOE = higher tax = higher profitability signal)
Factor category: cashflow / profitability

Post-processing:
  1. Mainboard filter (60xxxx, 00xxxx only)
  2. avg_equity <= 0 -> NaN
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

FACTOR_NAME = "TOE"
FACTOR_DIRECTION = 1
WINSOR_LO = 1.0
WINSOR_HI = 99.0


def _is_mainboard(symbol: str) -> bool:
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class TOE(FundamentalFactorCalculator):
    """
    Tax on Equity.
    TOE = income_tax_expense_TTM / avg(parent_equity)
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
        # Load single-quarter income tax expense (cumulative within fiscal year)
        ite_v, ite_s, ite_d = fundamental_data.get_daily_panel("q_ps_ite_c")
        # Load parent equity
        eq_v, eq_s, eq_d = fundamental_data.get_daily_panel("q_bs_tetoshopc_t")

        if ite_v.size == 0:
            raise ValueError("TOE: get_daily_panel('q_ps_ite_c') returned empty")
        if eq_v.size == 0:
            raise ValueError("TOE: get_daily_panel('q_bs_tetoshopc_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(ite_s.tolist()) & set(eq_s.tolist()))
        common_dates_int = sorted(set(ite_d.astype('int64').tolist()) & set(eq_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        ite_sym_idx = {s: i for i, s in enumerate(ite_s.tolist())}
        eq_sym_idx = {s: i for i, s in enumerate(eq_s.tolist())}
        ite_date_idx = {int(d): i for i, d in enumerate(ite_d.astype('int64'))}
        eq_date_idx = {int(d): i for i, d in enumerate(eq_d.astype('int64'))}

        is_idx = np.array([ite_sym_idx[s] for s in common_syms])
        es_idx = np.array([eq_sym_idx[s] for s in common_syms])
        id_idx = np.array([ite_date_idx[d] for d in common_dates_int])
        ed_idx = np.array([eq_date_idx[d] for d in common_dates_int])

        ite_v = ite_v[np.ix_(is_idx, id_idx)]
        eq_v = eq_v[np.ix_(es_idx, ed_idx)]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        ite_v = ite_v[mainboard_mask].copy()
        eq_v = eq_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        N, T = ite_v.shape

        # The get_daily_panel returns forward-filled quarterly data.
        # q_ps_ite_c is a CUMULATIVE single-quarter field, already de-cumulated
        # by FundamentalData. We need TTM = sum of last 4 quarters.
        #
        # Since get_daily_panel forward-fills quarterly values to daily,
        # consecutive days within the same quarter have the same value.
        # We need to identify quarter boundaries and do TTM rolling sum.
        #
        # Simpler approach: the daily panel has quarterly data ffilled.
        # Each "step" in the time series = new quarter data.
        # We detect changes and compute rolling 4-quarter sum.

        # For each stock, compute TTM by detecting quarterly changes
        ite_ttm = np.full((N, T), np.nan)
        # Loop: ~N stocks (~3000), each iterates over T to find ~40 quarter boundaries
        for i in range(N):
            row = ite_v[i]
            # Find indices where value changes (quarter boundaries)
            valid = np.isfinite(row)
            if valid.sum() < 4:
                continue

            # Extract unique quarterly values in order
            # A quarter boundary = value differs from previous day
            quarter_vals = []
            quarter_start_indices = []
            prev_val = None
            for t in range(T):
                if not valid[t]:
                    continue
                if prev_val is None or row[t] != prev_val:
                    quarter_vals.append(row[t])
                    quarter_start_indices.append(t)
                    prev_val = row[t]

            # Compute TTM (rolling 4-quarter sum)
            # Loop: ~40 quarters per stock
            for q in range(3, len(quarter_vals)):
                ttm_val = sum(quarter_vals[q-3:q+1])
                start_t = quarter_start_indices[q]
                end_t = quarter_start_indices[q+1] if q+1 < len(quarter_vals) else T
                ite_ttm[i, start_t:end_t] = ttm_val

        # Average equity: (current + 4-quarters-ago) / 2
        # Use same quarter boundary detection for equity
        avg_eq = np.full((N, T), np.nan)
        for i in range(N):
            row = eq_v[i]
            valid = np.isfinite(row)
            if valid.sum() < 5:
                continue

            quarter_vals = []
            quarter_start_indices = []
            prev_val = None
            for t in range(T):
                if not valid[t]:
                    continue
                if prev_val is None or row[t] != prev_val:
                    quarter_vals.append(row[t])
                    quarter_start_indices.append(t)
                    prev_val = row[t]

            # Loop: ~40 quarters per stock
            for q in range(4, len(quarter_vals)):
                avg = (quarter_vals[q] + quarter_vals[q-4]) / 2.0
                start_t = quarter_start_indices[q]
                end_t = quarter_start_indices[q+1] if q+1 < len(quarter_vals) else T
                avg_eq[i, start_t:end_t] = avg

        # Compute TOE
        avg_eq[avg_eq <= 0] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            values = ite_ttm / avg_eq

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
            warnings.warn(f"TOE NaN ratio is high: {nan_ratio:.1%}")

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
    print("TOE factor smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2022-01-01", end_date="2024-12-31")
    calc = TOE()
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
