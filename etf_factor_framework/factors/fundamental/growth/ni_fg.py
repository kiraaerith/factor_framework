"""
NI_FG factor (Net Income Forecast Growth Rate)

Computes:
  NI_FG = (ForecastNI_FY1 - ActualNI_t) / |ActualNI_t|

Data sources:
  - tushare.report_rc : np (10k CNY = 万元), quarter (e.g. '2025Q4')
                        -> 180-day rolling median analyst consensus for FY1
  - lixinger.financial_statements : q_ps_npatoshopc_c (yuan = 元, single-quarter)
                                    -> sum of Q1+Q2+Q3+Q4 for most recent disclosed fiscal year

FY1 definition:
  - FY1 = current calendar year (t.year), always.
  - Analysts predict current year throughout the year in tushare data.

Unit alignment:
  - tushare np is in 万元 -> multiply by 1e4 to convert to 元
  - lixinger q_ps_npatoshopc_c is in 元

Winsorization: factor values clipped to [-1, 10] before returning.
Minimum analyst count: require >= 2 analysts in 180-day window (else NaN).

Factor direction: positive (higher expected NI growth is better)
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

FACTOR_NAME = "NI_FG"
FACTOR_DIRECTION = 1  # positive: higher forecast NI growth is better
MIN_ANALYST_COUNT = 2  # minimum analysts required for consensus to be valid


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class NI_FG(FundamentalFactorCalculator):
    """
    Net Income Forecast Growth Rate.

    NI_FG = (ForecastNI_FY1 - ActualNI_t) / |ActualNI_t|

    ForecastNI_FY1: median analyst consensus NI for FY1 (tushare, 万元 -> 元)
    ActualNI_t: sum of single-quarter NI over 4 quarters for most recent
                disclosed fiscal year (lixinger, 元)
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
        """Compute NI_FG daily panel."""
        fd = fundamental_data
        fd._load_raw_data()
        trading_dates = fd._get_trading_dates()

        if fd._raw_data is None or fd._raw_data.empty:
            raise ValueError("NI_FG: lixinger raw data is empty")

        T = len(trading_dates)
        dates_arr = np.array(trading_dates, dtype='datetime64[ns]')

        # Step 1: Build actual annual NI panel from lixinger (元)
        actual_ni_panel, lx_symbols = self._load_actual_ni_panel(fd, trading_dates)

        # Step 2: Filter to mainboard
        mb_mask = np.array([_is_mainboard(s) for s in lx_symbols])
        lx_symbols = lx_symbols[mb_mask]
        actual_ni_panel = actual_ni_panel[mb_mask]

        if len(lx_symbols) == 0:
            raise ValueError("NI_FG: no mainboard symbols after filtering")

        # Step 3: Load FY1 analyst consensus panel from tushare (元)
        forecast_ni_panel = self._load_forecast_ni_panel(fd, lx_symbols, trading_dates)

        # Step 4: Compute growth rate
        with np.errstate(divide='ignore', invalid='ignore'):
            growth = (forecast_ni_panel - actual_ni_panel) / np.abs(actual_ni_panel)

        # Mask invalid cases: ActualNI == 0, either is NaN
        bad = (
            (actual_ni_panel == 0)
            | np.isnan(actual_ni_panel)
            | np.isnan(forecast_ni_panel)
            | np.isinf(growth)
        )
        growth[bad] = np.nan

        # Step 5: Winsorize to [-1, 10]
        finite_mask = ~np.isnan(growth)
        growth[finite_mask] = np.clip(growth[finite_mask], -1.0, 10.0)

        nan_ratio = np.isnan(growth).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"NI_FG: NaN ratio is high ({nan_ratio:.1%}). "
                "This is expected if few stocks have analyst coverage."
            )

        return FactorData(
            values=growth,
            symbols=lx_symbols,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal: actual annual NI from lixinger
    # ------------------------------------------------------------------

    def _load_actual_ni_panel(
        self,
        fd: FundamentalData,
        trading_dates: pd.DatetimeIndex,
    ):
        """
        Build daily actual annual NI panel from lixinger financial_statements.

        Annual NI = sum of single-quarter values Q1+Q2+Q3+Q4 for a fiscal year.
        The effective disclosure date is the Q4 report_date (max report_date
        among Q4 rows for that fiscal year). Values are forward-filled from
        that date onward.

        Returns:
            tuple: (values ndarray (N, T) float64 in yuan, symbols ndarray (N,))
        """
        field = 'q_ps_npatoshopc_c'
        df = fd._raw_data.copy()
        df = df[df[field].notna()].copy()

        if df.empty:
            return (
                np.empty((0, len(trading_dates)), dtype=np.float64),
                np.array([], dtype=str),
            )

        # Fiscal year from quarter end date
        df['fiscal_year'] = df['date'].dt.year

        # Sum 4 quarters per (symbol, fiscal_year) -> annual NI
        annual_ni = (
            df.groupby(['symbol', 'fiscal_year'])[field]
            .sum()
            .reset_index()
            .rename(columns={field: 'annual_ni'})
        )

        # Q4 report_date: max report_date among December-end rows
        q4_df = df[df['date'].dt.month == 12].copy()
        if q4_df.empty:
            return (
                np.empty((0, len(trading_dates)), dtype=np.float64),
                np.array([], dtype=str),
            )

        q4_disclosure = (
            q4_df.groupby(['symbol', 'fiscal_year'])['report_date']
            .max()
            .reset_index()
            .rename(columns={'report_date': 'disclosure_date'})
        )

        merged = annual_ni.merge(q4_disclosure, on=['symbol', 'fiscal_year'], how='inner')
        merged = merged[merged['disclosure_date'].notna()]
        merged = merged[merged['disclosure_date'] <= fd.end_date]

        if merged.empty:
            return (
                np.empty((0, len(trading_dates)), dtype=np.float64),
                np.array([], dtype=str),
            )

        # Pivot: rows=symbol, cols=disclosure_date, values=annual_ni
        pivot = merged.pivot_table(
            index='symbol',
            columns='disclosure_date',
            values='annual_ni',
            aggfunc='last',
        )

        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1)
        panel = panel.reindex(columns=trading_dates)

        values = panel.values.astype(np.float64)
        symbols = np.array(panel.index.tolist())

        return values, symbols

    # ------------------------------------------------------------------
    # Internal: FY1 analyst consensus from tushare
    # ------------------------------------------------------------------

    def _load_forecast_ni_panel(
        self,
        fd: FundamentalData,
        symbols: np.ndarray,
        trading_dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Build daily FY1 analyst consensus NI panel from tushare.report_rc.

        Uses 180-day rolling window. FY1 year:
          - month >= 5: current year
          - month <  5: current year - 1

        Requires >= MIN_ANALYST_COUNT analysts in the window.
        np values converted from 万元 to 元 (* 1e4).

        Returns:
            ndarray (N, T) float64 in yuan (元)
        """
        N = len(symbols)
        T = len(trading_dates)
        result = np.full((N, T), np.nan, dtype=np.float64)

        if N == 0 or T == 0:
            return result

        # Query date range
        dates_pd = pd.DatetimeIndex(trading_dates)
        query_start = (dates_pd[0] - pd.Timedelta(days=181)).strftime('%Y%m%d')
        query_end = fd.end_date.strftime('%Y%m%d')

        try:
            conn = sqlite3.connect(TUSHARE_DB)
            fc_df = pd.read_sql_query(
                f"""
                SELECT ts_code, report_date, quarter, np
                FROM report_rc
                WHERE report_date BETWEEN '{query_start}' AND '{query_end}'
                  AND np IS NOT NULL
                  AND quarter LIKE '%Q4'
                """,
                conn,
            )
            conn.close()
        except Exception as exc:
            warnings.warn(f"NI_FG: failed to load tushare data: {exc}")
            return result

        if fc_df.empty:
            return result

        # Filter to mainboard
        fc_df['code6'] = fc_df['ts_code'].str[:6]
        mb = fc_df['code6'].apply(
            lambda c: c.startswith('60') or c.startswith('00')
        )
        fc_df = fc_df[mb].copy()
        if fc_df.empty:
            return result

        # Convert 万元 -> 元
        fc_df['np_yuan'] = fc_df['np'].astype(float) * 1e4

        # Parse report_date (TEXT YYYYMMDD) -> datetime
        fc_df['report_date_dt'] = pd.to_datetime(fc_df['report_date'], format='%Y%m%d')

        # Extract forecast year from quarter string
        fc_df['q_year'] = fc_df['quarter'].str[:4].astype(int)

        # Build code6 -> symbol-index mapping
        code6_to_idx = {}
        for i, sym in enumerate(symbols):
            m = re.search(r'(\d{6})', sym)
            if m:
                code6_to_idx[m.group(1)] = i

        # Sort by report_date for binary search
        fc_df = fc_df.sort_values('report_date_dt').reset_index(drop=True)
        fc_dates_arr = fc_df['report_date_dt'].values.astype('datetime64[D]')
        dates_d = np.array(trading_dates, dtype='datetime64[D]')

        for t_idx, t in enumerate(dates_d):
            t_pd = pd.Timestamp(t)

            # FY1 year
            fy1_year = t_pd.year  # FY1 = current calendar year, always

            # 180-day window boundary via binary search
            t_180 = t - np.timedelta64(180, 'D')
            s_i = int(np.searchsorted(fc_dates_arr, t_180, side='left'))
            e_i = int(np.searchsorted(
                fc_dates_arr, t + np.timedelta64(1, 'D'), side='left'
            ))

            if s_i >= e_i:
                continue

            window = fc_df.iloc[s_i:e_i]
            fy1_window = window[window['q_year'] == fy1_year]
            if fy1_window.empty:
                continue

            # Median per stock; only include if >= MIN_ANALYST_COUNT reports
            grp = fy1_window.groupby('code6')['np_yuan']
            consensus_median = grp.median()
            consensus_count = grp.count()

            for code6, np_val in consensus_median.items():
                if consensus_count[code6] < MIN_ANALYST_COUNT:
                    continue
                if code6 in code6_to_idx:
                    result[code6_to_idx[code6], t_idx] = np_val

        return result


# ------------------------------------------------------------------
# Smoke test (run: python ni_fg.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("NI_FG factor smoke test")
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

    print(f"\n[Step 2] Compute NI_FG factor")
    calculator = NI_FG()
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
