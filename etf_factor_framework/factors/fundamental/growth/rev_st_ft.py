"""
REV_ST_FT factor (Revenue Short-Term Forecast Trend)

Computes the revision momentum of analyst FY1 operating revenue consensus:
  REV_ST_FT_1Q = (FY1_consensus(t) - FY1_consensus(t-65))  / |FY1_consensus(t-65)|
  REV_ST_FT_6M = (FY1_consensus(t) - FY1_consensus(t-130)) / |FY1_consensus(t-130)|

  REV_ST_FT = 0.5 * rank(REV_ST_FT_1Q) + 0.5 * rank(REV_ST_FT_6M)

FY1 definition:
  - FY1 = current calendar year (t.year), always.
  - For growth rate computation, t and t-window both use t.year as FY1 target.
  - No cross-year NaN check needed (analysts predict current year throughout).

Data source:
  - tushare.report_rc : op_rt (万元), quarter, report_date, org_name, ts_code
  - 180-day rolling window for point-in-time consensus
  - Minimum 2 analysts required for consensus

Post-processing per window:
  1. Winsorize to [-1, 5]
  2. Additional +-3sigma winsorization per cross-section
  3. Cross-section rank normalization (0~1)
  Composite: equal weight of the two window ranks.
  If only one window valid, use that window's rank directly.

Factor direction: positive (upward revision of revenue forecasts is better)
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
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # etf_cross_ml-master
TUSHARE_DB = str(_PROJECT_ROOT.parent / "china_stock_data" / "tushare.db")

FACTOR_NAME = "REV_ST_FT"
FACTOR_DIRECTION = 1  # positive: upward forecast revision is better

WINDOW_1Q = 65    # ~1 quarter in trading days
WINDOW_6M = 130   # ~6 months in trading days
MIN_ANALYST_COUNT = 2
CONSENSUS_WINDOW_DAYS = 180  # rolling look-back days for consensus


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _fy1_year(date: pd.Timestamp) -> int:
    """FY1 target year: always current calendar year."""
    return date.year


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


def _cross_section_rank(panel: np.ndarray) -> np.ndarray:
    """
    Cross-section rank normalization per date: rank each valid value to [0, 1].
    panel: (N, T) float64. NaN positions remain NaN.
    Returns (N, T) float64.
    """
    out = np.full_like(panel, np.nan)
    N, T = panel.shape
    for t in range(T):
        cs = panel[:, t]
        valid_mask = ~np.isnan(cs)
        n_valid = valid_mask.sum()
        if n_valid < 2:
            if n_valid == 1:
                out[valid_mask, t] = 0.5
            continue
        ranks = rankdata(cs[valid_mask], method='average')
        # Normalize to [0, 1]
        out[valid_mask, t] = (ranks - 1) / (n_valid - 1)
    return out


class REV_ST_FT(FundamentalFactorCalculator):
    """
    Revenue Short-Term Forecast Trend factor.

    Measures the momentum of analyst FY1 operating revenue consensus revisions
    over two windows (1Q = 65 trading days, 6M = 130 trading days).
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
            "window_1q": WINDOW_1Q,
            "window_6m": WINDOW_6M,
            "min_analyst_count": MIN_ANALYST_COUNT,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """Compute REV_ST_FT daily panel."""
        fd = fundamental_data
        fd._load_raw_data()
        trading_dates = fd._get_trading_dates()

        T = len(trading_dates)
        dates_arr = np.array(trading_dates, dtype='datetime64[ns]')

        # Step 1: Build mainboard symbol universe from tushare
        symbols = self._get_mainboard_symbols(fd)
        N = len(symbols)

        if N == 0:
            raise ValueError("REV_ST_FT: no mainboard symbols found in tushare data")

        # Step 2: Build FY1 consensus panel from tushare (万元)
        consensus_panel = self._build_consensus_panel(fd, symbols, trading_dates)
        # consensus_panel: (N, T) float64, units = 万元

        # Step 3: Compute growth rates for both windows
        growth_1q = np.full((N, T), np.nan, dtype=np.float64)
        growth_6m = np.full((N, T), np.nan, dtype=np.float64)

        for t_idx in range(T):
            curr = consensus_panel[:, t_idx]

            # 1Q window
            t0 = t_idx - WINDOW_1Q
            if t0 >= 0:
                prev = consensus_panel[:, t0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    g = (curr - prev) / np.abs(prev)
                bad = (prev == 0) | np.isnan(prev) | np.isnan(curr) | np.isinf(g)
                g = np.where(bad, np.nan, g)
                growth_1q[:, t_idx] = g

            # 6M window
            t0 = t_idx - WINDOW_6M
            if t0 >= 0:
                prev = consensus_panel[:, t0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    g = (curr - prev) / np.abs(prev)
                bad = (prev == 0) | np.isnan(prev) | np.isnan(curr) | np.isinf(g)
                g = np.where(bad, np.nan, g)
                growth_6m[:, t_idx] = g

        # Step 5: Winsorize each window: [-1, 5] then +-3sigma per cross-section
        growth_1q = _winsorize_panel(growth_1q, lower=-1.0, upper=5.0)
        growth_6m = _winsorize_panel(growth_6m, lower=-1.0, upper=5.0)

        # Step 6: Cross-section rank normalization (0~1) per window
        rank_1q = _cross_section_rank(growth_1q)
        rank_6m = _cross_section_rank(growth_6m)

        # Step 7: Composite
        both_valid  = ~np.isnan(rank_1q) & ~np.isnan(rank_6m)
        only_1q     = ~np.isnan(rank_1q) &  np.isnan(rank_6m)
        only_6m     =  np.isnan(rank_1q) & ~np.isnan(rank_6m)

        composite = np.full((N, T), np.nan, dtype=np.float64)
        composite[both_valid] = 0.5 * rank_1q[both_valid] + 0.5 * rank_6m[both_valid]
        composite[only_1q]    = rank_1q[only_1q]
        composite[only_6m]    = rank_6m[only_6m]

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"REV_ST_FT: NaN ratio is high ({nan_ratio:.1%}). "
                "Expected if few stocks have analyst coverage."
            )

        return FactorData(
            values=composite,
            symbols=symbols,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mainboard_symbols(self, fd: FundamentalData) -> np.ndarray:
        """
        Extract mainboard symbols from tushare report_rc (the actual data source).

        Using lixinger _raw_data would make the universe depend on fd.end_date,
        causing cross-section rank differences between full and truncated datasets
        (a false leakage signal). Querying tushare directly gives a stable universe
        independent of end_date.
        """
        try:
            conn = sqlite3.connect(TUSHARE_DB)
            df = pd.read_sql_query(
                "SELECT DISTINCT ts_code FROM report_rc"
                " WHERE quarter LIKE '%Q4' AND op_rt IS NOT NULL",
                conn,
            )
            conn.close()
        except Exception as exc:
            warnings.warn(f"REV_ST_FT: cannot load tushare symbols: {exc}")
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

    def _build_consensus_panel(
        self,
        fd: FundamentalData,
        symbols: np.ndarray,
        trading_dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Build daily FY1 analyst consensus revenue panel from tushare.report_rc.

        For each trading date t:
          - FY1 year = t.year (current calendar year, always)
          - Look at reports with report_date in [t - 180 days, t]
            AND quarter == f"{fy1_year}Q4"
          - Per org_name, take the most recent report
          - Require >= MIN_ANALYST_COUNT analysts; take median of op_rt (万元)

        Returns:
            ndarray (N, T) float64, units = 万元. NaN where insufficient coverage.
        """
        N = len(symbols)
        T = len(trading_dates)
        result = np.full((N, T), np.nan, dtype=np.float64)

        if N == 0 or T == 0:
            return result

        # Determine query date range (extend backward by CONSENSUS_WINDOW_DAYS)
        dates_pd = pd.DatetimeIndex(trading_dates)
        query_start = (
            dates_pd[0] - pd.Timedelta(days=CONSENSUS_WINDOW_DAYS + 5)
        ).strftime('%Y%m%d')
        query_end = fd.end_date.strftime('%Y%m%d')

        try:
            conn = sqlite3.connect(TUSHARE_DB)
            fc_df = pd.read_sql_query(
                f"""
                SELECT ts_code, report_date, quarter, org_name, op_rt
                FROM report_rc
                WHERE report_date BETWEEN '{query_start}' AND '{query_end}'
                  AND op_rt IS NOT NULL
                  AND quarter LIKE '%Q4'
                """,
                conn,
            )
            conn.close()
        except Exception as exc:
            warnings.warn(f"REV_ST_FT: failed to load tushare data: {exc}")
            return result

        if fc_df.empty:
            return result

        # Filter to mainboard stocks only
        fc_df['code6'] = fc_df['ts_code'].str[:6]
        mb_mask = fc_df['code6'].apply(
            lambda c: c.startswith('60') or c.startswith('00')
        )
        fc_df = fc_df[mb_mask].copy()
        if fc_df.empty:
            return result

        # Parse dates and extract forecast year
        fc_df['report_date_dt'] = pd.to_datetime(fc_df['report_date'], format='%Y%m%d')
        fc_df['q_year'] = fc_df['quarter'].str[:4].astype(int)
        fc_df['op_rt_val'] = fc_df['op_rt'].astype(float)

        # Build code6 -> symbol-index mapping
        code6_to_idx: dict = {}
        for i, sym in enumerate(symbols):
            m = re.search(r'(\d{6})', sym)
            if m:
                code6_to_idx[m.group(1)] = i

        # Sort by report_date + deterministic tie-breakers to ensure identical
        # results regardless of SQL row order (which varies with query size).
        fc_df = fc_df.sort_values(
            ['report_date_dt', 'ts_code', 'org_name', 'op_rt_val']
        ).reset_index(drop=True)
        fc_dates_arr = fc_df['report_date_dt'].values.astype('datetime64[D]')
        dates_d = np.array(trading_dates, dtype='datetime64[D]')

        for t_idx, t in enumerate(dates_d):
            t_pd = pd.Timestamp(t)
            fy1_year = _fy1_year(t_pd)

            # Rolling window: [t - 180 days, t]
            t_start = t - np.timedelta64(CONSENSUS_WINDOW_DAYS, 'D')
            s_i = int(np.searchsorted(fc_dates_arr, t_start, side='left'))
            e_i = int(np.searchsorted(
                fc_dates_arr, t + np.timedelta64(1, 'D'), side='left'
            ))

            if s_i >= e_i:
                continue

            window = fc_df.iloc[s_i:e_i]
            fy1_window = window[window['q_year'] == fy1_year]
            if fy1_window.empty:
                continue

            # Per org_name, keep only the most recent report.
            # fc_df is already sorted by (report_date_dt, ts_code, org_name, op_rt_val)
            # so .last() is deterministic even when report_date_dt ties exist.
            latest_per_org = (
                fy1_window
                .groupby(['code6', 'org_name'])['op_rt_val']
                .last()
            )

            # Consensus: median of orgs' latest op_rt, require >= MIN_ANALYST_COUNT
            grp = latest_per_org.groupby(level='code6')
            consensus_median = grp.median()
            consensus_count  = grp.count()

            for code6, rev_val in consensus_median.items():
                if consensus_count[code6] < MIN_ANALYST_COUNT:
                    continue
                if code6 in code6_to_idx:
                    result[code6_to_idx[code6], t_idx] = rev_val

        return result


# ------------------------------------------------------------------
# Smoke test (run: python rev_st_ft.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("REV_ST_FT factor smoke test")
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

    print(f"\n[Step 2] Compute REV_ST_FT factor")
    calculator = REV_ST_FT()
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
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

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
