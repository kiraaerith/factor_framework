"""
NI_LS_FD factor (Net Income Long-Short Forecast Divergence)

Measures the divergence between long-term (FY3) and short-term (FY1) analyst
earnings revision over a 1-year (252 trading-day) look-back window:

  NI_ST_FT_1Y(t) = (FY1_consensus(t) - FY1_consensus(t-252)) / |FY1_consensus(t-252)|
  NI_LT_FT_1Y(t) = (FY3_consensus(t) - FY3_consensus(t-252)) / |FY3_consensus(t-252)|

  NI_LS_FD = NI_LT_FT_1Y - NI_ST_FT_1Y

where FY1_consensus(t) uses fy1_year(t) as target and FY1_consensus(t-252) uses
fy1_year(t-252) as target (each timestamp uses its own current FY1 year).
No cross-year boundary check is applied: a positive reading means long-term
expectations grew faster than short-term over the past year, even if the
reference fiscal years differ by one.

FY1 definition (consistent with NI_ST_FT):
  - month >= 5: FY1 = current calendar year
  - month <  5: FY1 = current calendar year - 1

FY3 definition:
  - FY3 = FY1_year + 2   (quarter filter: f"{fy1_year + 2}Q4")

Post-processing:
  1. Winsorize each component to [-1, 5], then +-3sigma per cross-section
  2. Compute difference: NI_LS_FD = NI_LT_FT_1Y - NI_ST_FT_1Y
  3. Winsorize NI_LS_FD: +-3sigma per cross-section
  4. NaN if either component is NaN

Data source:
  - tushare.report_rc : np (万元), quarter, report_date, org_name, ts_code
  - 180-day rolling window for point-in-time consensus
  - Minimum 2 analysts required for consensus

Factor direction: positive (long-term revision outpacing short-term is better)
Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only
"""

import os
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # etf_cross_ml-master
TUSHARE_DB = str(_PROJECT_ROOT.parent / "china_stock_data" / "tushare.db")

FACTOR_NAME = "NI_LS_FD"
FACTOR_DIRECTION = 1  # positive: long-term revision outpacing short-term is better

WINDOW_1Y = 252         # ~1 year in trading days
MIN_ANALYST_COUNT = 2
CONSENSUS_WINDOW_DAYS = 180  # rolling look-back days for point-in-time consensus


def _fy1_year(date: pd.Timestamp) -> int:
    """FY1 target year: current year if month>=5, else current year - 1."""
    return date.year if date.month >= 5 else date.year - 1


def _winsorize_panel(panel: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Winsorize each cross-section to [lower, upper], then apply +-3sigma per cross-section.
    panel: (N, T) float64, NaN-safe.
    Returns a new (N, T) array.
    """
    out = panel.copy()
    T = out.shape[1]
    for t in range(T):
        cs = out[:, t]
        valid_mask = ~np.isnan(cs)
        if valid_mask.sum() < 2:
            continue
        cs = np.where(valid_mask, np.clip(cs, lower, upper), cs)
        valid_vals = cs[valid_mask]
        mu = valid_vals.mean()
        sigma = valid_vals.std()
        if sigma > 0:
            cs = np.where(
                valid_mask,
                np.clip(cs, mu - 3 * sigma, mu + 3 * sigma),
                cs,
            )
        out[:, t] = cs
    return out


def _winsorize_panel_sigma(panel: np.ndarray) -> np.ndarray:
    """Apply +-3sigma winsorization per cross-section (no hard bounds)."""
    out = panel.copy()
    T = out.shape[1]
    for t in range(T):
        cs = out[:, t]
        valid_mask = ~np.isnan(cs)
        if valid_mask.sum() < 2:
            continue
        valid_vals = cs[valid_mask]
        mu = valid_vals.mean()
        sigma = valid_vals.std()
        if sigma > 0:
            cs = np.where(
                valid_mask,
                np.clip(cs, mu - 3 * sigma, mu + 3 * sigma),
                cs,
            )
        out[:, t] = cs
    return out


class NI_LS_FD(FundamentalFactorCalculator):
    """
    Net Income Long-Short Forecast Divergence factor.

    Computes the difference between long-term (FY3) and short-term (FY1)
    analyst earnings revision over a 1-year (252 trading-day) window.
    Each date uses its own current FY1/FY3 year for both the current lookup
    and the historical (t-252) lookup (each with its own FY year).
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "direction": FACTOR_DIRECTION,
            "window_1y": WINDOW_1Y,
            "min_analyst_count": MIN_ANALYST_COUNT,
        }

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """Compute NI_LS_FD daily panel."""
        fd = fundamental_data
        fd._load_raw_data()
        trading_dates = fd._get_trading_dates()

        T = len(trading_dates)
        dates_arr = np.array(trading_dates, dtype='datetime64[ns]')

        # Step 1: Build mainboard symbol universe from tushare
        symbols = self._get_mainboard_symbols()
        N = len(symbols)

        if N == 0:
            raise ValueError("NI_LS_FD: no mainboard symbols found in tushare data")

        # Step 2: Build FY1 and FY3 consensus panels from tushare
        # Each date t uses fy1_year(t) as the target fiscal year.
        fy1_panel, fy3_panel = self._build_consensus_panels(
            fd, symbols, trading_dates
        )
        # fy1_panel, fy3_panel: (N, T) float64, units = 万元

        # Step 3: Compute 1-year growth rates
        # growth_st[t] = (fy1_panel[t] - fy1_panel[t-252]) / |fy1_panel[t-252]|
        # growth_lt[t] = (fy3_panel[t] - fy3_panel[t-252]) / |fy3_panel[t-252]|
        # No FY-year equality check: each timestamp uses its own FY1/FY3 year.
        growth_st = np.full((N, T), np.nan, dtype=np.float64)
        growth_lt = np.full((N, T), np.nan, dtype=np.float64)

        for t_idx in range(WINDOW_1Y, T):
            t0 = t_idx - WINDOW_1Y

            # Short-term (FY1) growth
            curr_fy1 = fy1_panel[:, t_idx]
            prev_fy1 = fy1_panel[:, t0]
            with np.errstate(divide='ignore', invalid='ignore'):
                g_st = (curr_fy1 - prev_fy1) / np.abs(prev_fy1)
            bad = (prev_fy1 == 0) | np.isnan(prev_fy1) | np.isnan(curr_fy1) | np.isinf(g_st)
            growth_st[:, t_idx] = np.where(bad, np.nan, g_st)

            # Long-term (FY3) growth
            curr_fy3 = fy3_panel[:, t_idx]
            prev_fy3 = fy3_panel[:, t0]
            with np.errstate(divide='ignore', invalid='ignore'):
                g_lt = (curr_fy3 - prev_fy3) / np.abs(prev_fy3)
            bad = (prev_fy3 == 0) | np.isnan(prev_fy3) | np.isnan(curr_fy3) | np.isinf(g_lt)
            growth_lt[:, t_idx] = np.where(bad, np.nan, g_lt)

        # Step 4: Winsorize each component: [-1, 5] then +-3sigma per cross-section
        growth_st = _winsorize_panel(growth_st, lower=-1.0, upper=5.0)
        growth_lt = _winsorize_panel(growth_lt, lower=-1.0, upper=5.0)

        # Step 5: NI_LS_FD = NI_LT_FT_1Y - NI_ST_FT_1Y (NaN if either missing)
        diff = np.where(
            np.isnan(growth_st) | np.isnan(growth_lt),
            np.nan,
            growth_lt - growth_st,
        )

        # Step 6: Winsorize diff: +-3sigma per cross-section
        diff = _winsorize_panel_sigma(diff)

        nan_ratio = np.isnan(diff).mean()
        if nan_ratio > 0.95:
            warnings.warn(
                f"NI_LS_FD: NaN ratio is high ({nan_ratio:.1%}). "
                "FY3 coverage is lower than FY1; some NaN expected for 1-year window."
            )

        return FactorData(
            values=diff,
            symbols=symbols,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mainboard_symbols(self) -> np.ndarray:
        """Extract mainboard symbols from tushare report_rc (stable universe)."""
        try:
            conn = sqlite3.connect(TUSHARE_DB)
            df = pd.read_sql_query(
                "SELECT DISTINCT ts_code FROM report_rc"
                " WHERE quarter LIKE '%Q4' AND np IS NOT NULL",
                conn,
            )
            conn.close()
        except Exception as exc:
            warnings.warn(f"NI_LS_FD: cannot load tushare symbols: {exc}")
            return np.array([], dtype=str)

        symbols = []
        for ts_code in df['ts_code']:
            m = re.search(r'(\d{6})', ts_code)
            if not m:
                continue
            code6 = m.group(1)
            if code6.startswith('60'):
                symbols.append(f"SHSE.{code6}")
            elif code6.startswith('00'):
                symbols.append(f"SZSE.{code6}")

        return np.array(sorted(set(symbols)))

    def _build_consensus_panels(
        self,
        fd: FundamentalData,
        symbols: np.ndarray,
        trading_dates: pd.DatetimeIndex,
    ) -> tuple:
        """
        Build daily FY1 and FY3 analyst consensus NI panels from tushare.report_rc.

        For each trading date t:
          - FY1 year = fy1_year(t) = t.year if t.month>=5 else t.year-1
          - FY3 year = FY1 year + 2
          - Look at reports with report_date in [t - 180 days, t]
            filtered to q_year == FY1 year (or FY3 year respectively)
          - Per org_name per stock, keep the most recent report
          - Require >= MIN_ANALYST_COUNT analysts; take median of np (万元)

        Returns:
            (fy1_panel, fy3_panel): each (N, T) float64, units = 万元.
        """
        N = len(symbols)
        T = len(trading_dates)
        fy1_result = np.full((N, T), np.nan, dtype=np.float64)
        fy3_result = np.full((N, T), np.nan, dtype=np.float64)

        if N == 0 or T == 0:
            return fy1_result, fy3_result

        # Determine query date range
        dates_pd = pd.DatetimeIndex(trading_dates)
        query_start = (
            dates_pd[0] - pd.Timedelta(days=CONSENSUS_WINDOW_DAYS + 5)
        ).strftime('%Y%m%d')
        query_end = fd.end_date.strftime('%Y%m%d')

        try:
            conn = sqlite3.connect(TUSHARE_DB)
            fc_df = pd.read_sql_query(
                f"""
                SELECT ts_code, report_date, quarter, org_name, np
                FROM report_rc
                WHERE report_date BETWEEN '{query_start}' AND '{query_end}'
                  AND np IS NOT NULL
                  AND quarter LIKE '%Q4'
                """,
                conn,
            )
            conn.close()
        except Exception as exc:
            warnings.warn(f"NI_LS_FD: failed to load tushare data: {exc}")
            return fy1_result, fy3_result

        if fc_df.empty:
            return fy1_result, fy3_result

        # Filter to mainboard stocks only
        fc_df['code6'] = fc_df['ts_code'].str[:6]
        mb_mask = fc_df['code6'].apply(
            lambda c: c.startswith('60') or c.startswith('00')
        )
        fc_df = fc_df[mb_mask].copy()
        if fc_df.empty:
            return fy1_result, fy3_result

        # Parse dates and extract forecast year
        fc_df['report_date_dt'] = pd.to_datetime(fc_df['report_date'], format='%Y%m%d')
        fc_df['q_year'] = fc_df['quarter'].str[:4].astype(int)
        fc_df['np_val'] = fc_df['np'].astype(float)

        # Build code6 -> symbol-index mapping
        code6_to_idx: dict = {}
        for i, sym in enumerate(symbols):
            m = re.search(r'(\d{6})', sym)
            if m:
                code6_to_idx[m.group(1)] = i

        # Sort deterministically for idempotency
        fc_df = fc_df.sort_values(
            ['report_date_dt', 'ts_code', 'org_name', 'np_val']
        ).reset_index(drop=True)
        fc_dates_arr = fc_df['report_date_dt'].values.astype('datetime64[D]')
        dates_d = np.array(trading_dates, dtype='datetime64[D]')

        for t_idx, t in enumerate(dates_d):
            t_pd = pd.Timestamp(t)
            fy1_yr = _fy1_year(t_pd)
            fy3_yr = fy1_yr + 2

            # Rolling window: [t - 180 days, t]
            t_start = t - np.timedelta64(CONSENSUS_WINDOW_DAYS, 'D')
            s_i = int(np.searchsorted(fc_dates_arr, t_start, side='left'))
            e_i = int(np.searchsorted(
                fc_dates_arr, t + np.timedelta64(1, 'D'), side='left'
            ))

            if s_i >= e_i:
                continue

            window = fc_df.iloc[s_i:e_i]

            # FY1 consensus
            fy1_window = window[window['q_year'] == fy1_yr]
            if not fy1_window.empty:
                latest_per_org = (
                    fy1_window
                    .groupby(['code6', 'org_name'])['np_val']
                    .last()
                )
                grp = latest_per_org.groupby(level='code6')
                consensus_median = grp.median()
                consensus_count  = grp.count()
                for code6, np_val in consensus_median.items():
                    if consensus_count[code6] < MIN_ANALYST_COUNT:
                        continue
                    if code6 in code6_to_idx:
                        fy1_result[code6_to_idx[code6], t_idx] = np_val

            # FY3 consensus
            fy3_window = window[window['q_year'] == fy3_yr]
            if not fy3_window.empty:
                latest_per_org = (
                    fy3_window
                    .groupby(['code6', 'org_name'])['np_val']
                    .last()
                )
                grp = latest_per_org.groupby(level='code6')
                consensus_median = grp.median()
                consensus_count  = grp.count()
                for code6, np_val in consensus_median.items():
                    if consensus_count[code6] < MIN_ANALYST_COUNT:
                        continue
                    if code6 in code6_to_idx:
                        fy3_result[code6_to_idx[code6], t_idx] = np_val

        return fy1_result, fy3_result


# ------------------------------------------------------------------
# Smoke test (run: python ni_ls_fd.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("NI_LS_FD factor smoke test")
    print("=" * 60)

    TEST_START = "2020-01-01"
    TEST_END   = "2024-12-31"
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print(f"\n[Step 2] Compute NI_LS_FD factor")
    calculator = NI_LS_FD()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.shape}")
    print(f"Symbols: {result.symbols}")
    print(
        f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ "
        f"{pd.Timestamp(result.dates[-1]).date()}"
    )

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    # -- Smoke test assertions --
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, (
        f"expected float64, got {result.values.dtype}"
    )
    # NI_LS_FD requires 252-day burn-in AND both FY1 & FY3 valid.
    # FY3 has lower coverage than FY1. Allow up to 95% NaN.
    assert nan_ratio < 0.95, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all(
        (result.values == result2.values) | both_nan
    ), "Idempotency failed"

    print(f"\n[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {last5}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid: {len(valid_cs)}")
        print(f"  mean: {valid_cs.mean():.4f}")
        print(f"  median: {np.median(valid_cs):.4f}")
        print(f"  min: {valid_cs.min():.4f}")
        print(f"  max: {valid_cs.max():.4f}")
    else:
        print("  (no valid values in last cross-section for test stocks)")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2013-01-01", end_date="2025-12-31", stock_codes=None)
    leakage_found = False
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n--- split_ratio={sr} ---")
        detector = FundamentalLeakageDetector(split_ratio=sr)
        report = detector.detect(calculator, fd_leak)
        report.print_report()
        if report.has_leakage:
            leakage_found = True
            print(f"[FAIL] Leakage detected at split_ratio={sr}")
        else:
            print(f"[OK] No leakage at split_ratio={sr}")

    if leakage_found:
        print("\n[RESULT] LEAKAGE DETECTED")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL PASSED - No leakage")
