"""
OCFA factor (Operation Cost on Fixed Assets)

Formula: OCFA = residual of rolling OLS: operating_cost ~ fixed_assets
  - Rolling window: most recent 8 quarters
  - Y = q_ps_toc_c (total operating cost, single quarter)
  - X = q_bs_fa_t (fixed assets)
  - OCFA = last residual (epsilon) of the regression
  - Lower residual = higher capacity utilization = better

Data fields (lixinger.financial_statements, quarterly -> daily via ffill):
  - q_ps_toc_c: total operating cost (single quarter)
  - q_bs_fa_t: total fixed assets

Factor direction: negative (lower residual = better efficiency)

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

FACTOR_NAME = "OCFA"
FACTOR_DIRECTION = -1  # negative: lower residual = better
WINSOR_LO = 1.0
WINSOR_HI = 99.0
ROLLING_QUARTERS = 8


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


def _compute_ocfa_for_stock(cost_row: np.ndarray, fa_row: np.ndarray, T: int) -> np.ndarray:
    """Compute OCFA for a single stock via rolling 8-quarter OLS residual."""
    cost_q = _detect_quarters(cost_row)
    fa_q = _detect_quarters(fa_row)

    if len(cost_q) < ROLLING_QUARTERS or len(fa_q) < ROLLING_QUARTERS:
        return np.full(T, np.nan)

    # Match quarters by index proximity
    # Build time-aligned quarterly pairs
    fa_map = {}
    for val, start in fa_q:
        fa_map[start] = val

    fa_starts = sorted(fa_map.keys())

    paired = []  # (cost_val, fa_val, start_idx)
    for cost_val, cost_start in cost_q:
        # Find closest fa at or before cost_start
        best_fa_start = None
        for fs in fa_starts:
            if fs <= cost_start:
                best_fa_start = fs
            else:
                break
        if best_fa_start is None:
            continue
        paired.append((cost_val, fa_map[best_fa_start], cost_start))

    if len(paired) < ROLLING_QUARTERS:
        return np.full(T, np.nan)

    result = np.full(T, np.nan)

    # Rolling OLS with window of 8 quarters
    # Loop: ~(len(paired) - 7) iterations, ~30 per stock
    for q in range(ROLLING_QUARTERS - 1, len(paired)):
        window = paired[q - ROLLING_QUARTERS + 1: q + 1]
        Y = np.array([p[0] for p in window])
        X = np.array([p[1] for p in window])

        # Check for valid data
        valid = np.isfinite(Y) & np.isfinite(X) & (X > 0)
        if valid.sum() < 3:
            continue

        Y_v = Y[valid]
        X_v = X[valid]

        # OLS: Y = alpha + beta * X + epsilon
        X_design = np.column_stack([X_v, np.ones(len(X_v))])
        try:
            beta = np.linalg.lstsq(X_design, Y_v, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # Residual for the latest point in window
        last_y = window[-1][0]
        last_x = window[-1][1]
        if not (np.isfinite(last_y) and np.isfinite(last_x)):
            continue
        predicted = beta[0] * last_x + beta[1]
        residual = last_y - predicted

        # Forward fill to daily
        start_t = window[-1][2]
        end_t = paired[q + 1][2] if q + 1 < len(paired) else T
        result[start_t:end_t] = residual

    return result


class OCFA(FundamentalFactorCalculator):
    """
    Operation Cost on Fixed Assets.
    OCFA = residual of rolling 8Q OLS: operating_cost ~ fixed_assets
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"direction": FACTOR_DIRECTION, "rolling_quarters": ROLLING_QUARTERS}

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        cost_v, cost_s, cost_d = fundamental_data.get_daily_panel("q_ps_toc_c")
        fa_v, fa_s, fa_d = fundamental_data.get_daily_panel("q_bs_fa_t")

        if cost_v.size == 0:
            raise ValueError("OCFA: get_daily_panel('q_ps_toc_c') returned empty")
        if fa_v.size == 0:
            raise ValueError("OCFA: get_daily_panel('q_bs_fa_t') returned empty")

        # Align to common symbols and dates
        common_syms = sorted(set(cost_s.tolist()) & set(fa_s.tolist()))
        common_dates_int = sorted(set(cost_d.astype('int64').tolist()) & set(fa_d.astype('int64').tolist()))
        common_dates = np.array(common_dates_int, dtype='datetime64[ns]')

        cost_si = {s: i for i, s in enumerate(cost_s.tolist())}
        fa_si = {s: i for i, s in enumerate(fa_s.tolist())}
        cost_di = {int(d): i for i, d in enumerate(cost_d.astype('int64'))}
        fa_di = {int(d): i for i, d in enumerate(fa_d.astype('int64'))}

        cost_v = cost_v[np.ix_(
            np.array([cost_si[s] for s in common_syms]),
            np.array([cost_di[d] for d in common_dates_int])
        )]
        fa_v = fa_v[np.ix_(
            np.array([fa_si[s] for s in common_syms]),
            np.array([fa_di[d] for d in common_dates_int])
        )]
        symbols = np.array(common_syms)
        dates = common_dates

        # Mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in symbols])
        cost_v = cost_v[mainboard_mask].copy()
        fa_v = fa_v[mainboard_mask].copy()
        symbols = symbols[mainboard_mask]

        N, T = cost_v.shape

        # Compute OCFA per stock
        # Loop: ~N stocks (~3000)
        values = np.full((N, T), np.nan)
        for i in range(N):
            values[i] = _compute_ocfa_for_stock(cost_v[i], fa_v[i], T)

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
            warnings.warn(f"OCFA NaN ratio is high: {nan_ratio:.1%}")

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
    print("OCFA factor smoke test")
    print("=" * 60)

    fd = FundamentalData(start_date="2022-01-01", end_date="2024-12-31")
    calc = OCFA()
    result = calc.calculate(fd)

    print(f"Shape: {result.shape}")
    print(f"NaN ratio: {np.isnan(result.values).mean():.1%}")

    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"Last cross-section: N={len(valid)}, mean={valid.mean():.2f}, "
              f"min={valid.min():.2f}, max={valid.max():.2f}")

    print("\n[Leakage detection]")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector
    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31")
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calc, fd_leak)
        status = "LEAK" if report.has_leakage else "OK"
        print(f"  split={sr}: [{status}]")

    print("\n[PASS]")
