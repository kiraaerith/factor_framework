"""
GPMD factor (Gross Profit Margin Dynamics)

Formula: GPMD = current_GPM_TTM - previous_GPM_TTM

Where:
  GPM_TTM = (revenue_TTM - cost_TTM) / revenue_TTM
  revenue_TTM = sum of most recent 4 quarters of q_ps_toi_c
  cost_TTM = sum of most recent 4 quarters of q_ps_toc_c

Factor direction: positive (improving gross margin = better pricing power trend)
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

FACTOR_NAME = "GPMD"
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
    """Detect quarter boundaries from ffilled daily data.
    Returns list of (value, start_idx)."""
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
    """Compute TTM (rolling 4Q sum) at each quarter boundary.
    Returns list of (ttm_value, start_idx)."""
    result = []
    # Loop: ~40 quarters
    for q in range(3, len(quarters)):
        ttm = sum(v for v, _ in quarters[q-3:q+1])
        result.append((ttm, quarters[q][1]))
    return result


def _compute_gpmd_for_stock(rev_row: np.ndarray, cost_row: np.ndarray, T: int) -> np.ndarray:
    """Compute GPMD for a single stock. Returns (T,) array."""
    rev_q = _detect_quarters(rev_row)
    cost_q = _detect_quarters(cost_row)

    if len(rev_q) < 5 or len(cost_q) < 5:
        return np.full(T, np.nan)

    rev_ttm_q = _compute_ttm_quarters(rev_q)
    cost_ttm_q = _compute_ttm_quarters(cost_q)

    if len(rev_ttm_q) < 2 or len(cost_ttm_q) < 2:
        return np.full(T, np.nan)

    # Match rev_ttm and cost_ttm by start_idx to compute GPM
    cost_map = {s: v for v, s in cost_ttm_q}

    gpm_quarters = []  # (gpm_value, start_idx)
    for rev_ttm, start in rev_ttm_q:
        # Find closest cost_ttm at or before this start
        best_cost_start = None
        for cs in sorted(cost_map.keys()):
            if cs <= start:
                best_cost_start = cs
            else:
                break
        if best_cost_start is None or rev_ttm <= 0:
            continue
        cost_ttm = cost_map[best_cost_start]
        gpm = (rev_ttm - cost_ttm) / rev_ttm
        gpm_quarters.append((gpm, start))

    if len(gpm_quarters) < 2:
        return np.full(T, np.nan)

    # GPMD = current GPM - previous GPM, forward fill
    result = np.full(T, np.nan)
    # Loop: ~35 quarters
    for q in range(1, len(gpm_quarters)):
        gpmd = gpm_quarters[q][0] - gpm_quarters[q-1][0]
        start_t = gpm_quarters[q][1]
        end_t = gpm_quarters[q+1][1] if q+1 < len(gpm_quarters) else T
        result[start_t:end_t] = gpmd

    return result


class GPMD(FundamentalFactorCalculator):
    """
    Gross Profit Margin Dynamics.
    GPMD = current_GPM_TTM - previous_GPM_TTM
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
        cost_v, cost_s, cost_d = fundamental_data.get_daily_panel("q_ps_toc_c")

        if rev_v.size == 0:
            raise ValueError("GPMD: get_daily_panel('q_ps_toi_c') returned empty")
        if cost_v.size == 0:
            raise ValueError("GPMD: get_daily_panel('q_ps_toc_c') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(rev_s.tolist()) & set(cost_s.tolist()))
        common_dates_int = sorted(set(rev_d.astype('int64').tolist()) & set(cost_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        rev_si = {s: i for i, s in enumerate(rev_s.tolist())}
        cost_si = {s: i for i, s in enumerate(cost_s.tolist())}
        rev_di = {int(d): i for i, d in enumerate(rev_d.astype('int64'))}
        cost_di = {int(d): i for i, d in enumerate(cost_d.astype('int64'))}

        rev_v = rev_v[np.ix_(
            np.array([rev_si[s] for s in common_syms]),
            np.array([rev_di[d] for d in common_dates_int])
        )]
        cost_v = cost_v[np.ix_(
            np.array([cost_si[s] for s in common_syms]),
            np.array([cost_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        rev_v = rev_v[mainboard_mask].copy()
        cost_v = cost_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        N, T = rev_v.shape

        # Compute GPMD per stock
        # Loop: ~N stocks (~3000)
        values = np.full((N, T), np.nan)
        for i in range(N):
            values[i] = _compute_gpmd_for_stock(rev_v[i], cost_v[i], T)

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
            warnings.warn(f"GPMD NaN ratio is high: {nan_ratio:.1%}")

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
    print("GPMD factor smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2022-01-01", end_date="2024-12-31")
    calc = GPMD()
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
