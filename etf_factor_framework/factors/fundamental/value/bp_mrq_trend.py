"""
BP_MRQ_TREND factor (BP MRQ 3-Year Time-Series Trend, Purified from 3-Year Reversal)

Formula:
  1. BP_MRQ = 1 / pb  (daily, from lixinger.fundamental)
     NaN if |pb| < 1e-6; retain negative pb (negative BV stocks)
  2. TREND_t = (BP_MRQ_t - BP_MRQ_{t-756}) / |BP_MRQ_{t-756}|
               per stock, 756-trading-day lookback (shift-based, not rolling)
               NaN if |BP_MRQ_{t-756}| < 1e-6 or BP_MRQ_{t-756} is NaN
  3. REV_3Y_t = close_t / close_{t-756} - 1  (from juejin ohlcv_adjusted)
  4. BP_MRQ_TREND = residual of cross-section OLS: TREND_t ~ z-score(REV_3Y_t)

Data fields:
  - pb    : lixinger.fundamental (daily valuation)
  - close : juejin ohlcv_adjusted (daily adjusted close price)

Factor direction: positive (higher TREND = BP improved more over 3Y,
                             net of price reversal = mean-reversion opportunity)
Factor category: value - time-series trend

Notes:
  - Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only.
  - Stocks listed < 3 years ago will have NaN TREND (insufficient history).
  - REV_3Y uses position-based shift(756) on extended price panel; the factor
    date aligns to trading dates, naturally avoiding same-day leakage.
  - PRICE_LOOKBACK_DAYS is set to load enough historical data before start_date
    to compute REV_3Y (756 trading days ~ 1200 calendar days).
"""

import os
import re
import sys
import warnings

import sqlite3
import numpy as np
import pandas as pd
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData, lixinger_code_to_symbol
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "BP_MRQ_TREND"
FACTOR_DIRECTION = 1       # positive: higher trend = BP improving faster = better value
MIN_ABS_PB = 1e-6          # near-zero PB threshold for NaN assignment
TREND_WINDOW = 756         # 3-year lookback in trading days
MIN_VALID_STOCKS = 30      # minimum cross-section size for OLS
PRICE_LOOKBACK_DAYS = 1200 # calendar days before start_date for extended data loading


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class BP_MRQ_TREND(FundamentalFactorCalculator):
    """
    BP MRQ 3-Year Time-Series Trend factor, purified from 3-year price reversal.

    Steps:
      1. BP_MRQ = 1/pb (lixinger.fundamental, daily)
      2. TREND = (BP_MRQ_t - BP_MRQ_{t-756}) / |BP_MRQ_{t-756}|, per stock
      3. REV_3Y = 756-trading-day cumulative return (juejin ohlcv_adjusted)
      4. BP_MRQ_TREND = residual of cross-section OLS: TREND ~ z-score(REV_3Y), per day

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
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
        """
        Calculate BP_MRQ_TREND daily panel.

        Returns:
            FactorData: N stocks x T days, values = OLS residuals (float64)
        """
        # ------------------------------------------------------------------
        # Step 1: Load pb, compute BP_MRQ, apply mainboard filter
        # ------------------------------------------------------------------
        pb_values, symbols, dates = fundamental_data.get_valuation_panel("pb")

        mb_mask = np.array([_is_mainboard(s) for s in symbols])
        pb_values = pb_values[mb_mask]
        symbols = symbols[mb_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bp_values = np.where(
                np.abs(pb_values) < MIN_ABS_PB,
                np.nan,
                1.0 / pb_values,
            )
        bp_values = np.where(np.isinf(bp_values), np.nan, bp_values).astype(np.float64)

        trading_dates = pd.DatetimeIndex(dates)
        N = len(symbols)
        T = len(dates)

        if N == 0 or T == 0:
            return FactorData(
                values=np.empty((0, T), dtype=np.float64),
                symbols=symbols,
                dates=dates,
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ------------------------------------------------------------------
        # Step 2: Compute 3-year TREND per stock using shift(756)
        # TREND_t = (BP_MRQ_t - BP_MRQ_{t-756}) / |BP_MRQ_{t-756}|
        # We need extended PB history before start_date for the shift.
        # ------------------------------------------------------------------
        extended_start = (fundamental_data.start_date - pd.DateOffset(days=PRICE_LOOKBACK_DAYS)).date()
        load_end = fundamental_data.end_date.date()

        conn_lx = sqlite3.connect(fundamental_data._lixinger_db)
        df_pb_ext = pd.read_sql_query(
            f"""
            SELECT stock_code, date, pb
            FROM fundamental
            WHERE substr(date, 1, 10) BETWEEN '{extended_start}' AND '{load_end}'
              AND pb IS NOT NULL
            """,
            conn_lx,
        )
        conn_lx.close()

        if not df_pb_ext.empty:
            _d = pd.to_datetime(df_pb_ext["date"])
            df_pb_ext["date"] = _d.dt.tz_localize(None) if _d.dt.tz is not None else _d
            df_pb_ext["symbol"] = df_pb_ext["stock_code"].apply(lixinger_code_to_symbol)

        # Get extended trading calendar from tushare trade_cal
        tushare_db = fundamental_data._tushare_db
        ext_start_str = pd.Timestamp(extended_start).strftime("%Y%m%d")
        ext_end_str = pd.Timestamp(load_end).strftime("%Y%m%d")

        conn_ts = sqlite3.connect(tushare_db)
        ext_dates_df = pd.read_sql_query(
            f"""
            SELECT cal_date AS trade_date FROM trade_cal
            WHERE exchange = 'SSE' AND is_open = 1
              AND cal_date BETWEEN '{ext_start_str}' AND '{ext_end_str}'
            ORDER BY cal_date
            """, conn_ts
        )
        conn_ts.close()
        ext_trading_dates = pd.DatetimeIndex(pd.to_datetime(ext_dates_df["trade_date"]))

        # Build extended BP_MRQ panel (stocks x extended_dates)
        bp_trend_values = np.full((N, T), np.nan, dtype=np.float64)

        if not df_pb_ext.empty:
            # Filter mainboard only
            mb_mask_ext = df_pb_ext["symbol"].apply(_is_mainboard)
            df_pb_ext = df_pb_ext[mb_mask_ext]

            # Pivot: (symbol x extended_trading_dates)
            pb_ext_pivot = df_pb_ext.pivot_table(
                index="symbol", columns="date", values="pb", aggfunc="last"
            )
            # Merge with extended trading calendar and forward-fill
            all_ext_dates = pb_ext_pivot.columns.union(ext_trading_dates).sort_values()
            pb_ext_pivot = pb_ext_pivot.reindex(columns=all_ext_dates)
            pb_ext_panel = pb_ext_pivot.ffill(axis=1).reindex(columns=ext_trading_dates)

            # Compute BP_MRQ on extended panel
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bp_ext_panel = pb_ext_panel.copy()
                bp_ext_panel[:] = np.where(
                    np.abs(pb_ext_panel.values) < MIN_ABS_PB,
                    np.nan,
                    1.0 / pb_ext_panel.values,
                )
            bp_ext_panel = bp_ext_panel.where(~np.isinf(bp_ext_panel), np.nan)

            # Compute TREND = (BP_t - BP_{t-756}) / |BP_{t-756}|  using shift
            bp_lagged = bp_ext_panel.shift(TREND_WINDOW, axis=1)  # position-based shift

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                trend_ext = (bp_ext_panel - bp_lagged) / bp_lagged.abs()

            # Set NaN where lagged is near zero or NaN
            trend_ext = trend_ext.where(bp_lagged.abs() >= MIN_ABS_PB, np.nan)
            trend_ext = trend_ext.where(~np.isinf(trend_ext), np.nan)

            # Align to (symbols x trading_dates)
            trend_aligned = (
                trend_ext
                .reindex(index=symbols)
                .reindex(columns=trading_dates)
            )
            bp_trend_values = trend_aligned.values.astype(np.float64)

        # ------------------------------------------------------------------
        # Step 3: Load close prices from DuckDB, compute REV_3Y
        # ------------------------------------------------------------------
        conn_ts2 = sqlite3.connect(tushare_db)
        price_df = pd.read_sql_query(
            f"""
            SELECT ts_code, trade_date, close
            FROM daily_hfq
            WHERE (ts_code LIKE '60%' OR ts_code LIKE '00%')
              AND trade_date BETWEEN '{ext_start_str}' AND '{ext_end_str}'
              AND close IS NOT NULL
            """, conn_ts2
        )
        conn_ts2.close()

        def _ts_to_jq(ts_code):
            code, market = ts_code.split(".")
            return f"{'SHSE' if market == 'SH' else 'SZSE'}.{code}"

        if not price_df.empty:
            price_df["symbol"] = price_df["ts_code"].apply(_ts_to_jq)
        print(f"  - Price data: {len(price_df)} rows, {price_df['symbol'].nunique() if not price_df.empty else 0} stocks (tushare)")

        rev3y_values = np.full((N, T), np.nan, dtype=np.float64)

        if not price_df.empty:
            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

            close_panel = price_df.pivot_table(
                index="symbol", columns="trade_date", values="close", aggfunc="last"
            )
            close_panel = close_panel.reindex(columns=ext_trading_dates)
            close_panel = close_panel.ffill(axis=1)

            # REV_3Y = close_t / close_{t-756} - 1  (position-based shift)
            close_shifted = close_panel.shift(TREND_WINDOW, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rev3y_panel = close_panel / close_shifted - 1.0

            rev3y_aligned = (
                rev3y_panel
                .reindex(index=symbols)
                .reindex(columns=trading_dates)
            )
            rev3y_values = rev3y_aligned.values.astype(np.float64)

        # ------------------------------------------------------------------
        # Step 4: Cross-section OLS per day: TREND ~ z-score(REV_3Y), residuals
        # ------------------------------------------------------------------
        factor_values = bp_trend_values.copy()

        for t in range(T):
            trend_t = bp_trend_values[:, t]
            rev_t = rev3y_values[:, t]

            valid = ~np.isnan(trend_t) & ~np.isnan(rev_t)
            n_valid = int(valid.sum())

            if n_valid < MIN_VALID_STOCKS:
                factor_values[:, t] = np.nan
                continue

            trend_v = trend_t[valid]
            rev_v = rev_t[valid]

            # Z-score standardize REV_3Y cross-sectionally
            rev_std = rev_v.std()
            if rev_std < 1e-10:
                factor_values[:, t] = np.nan
                continue
            rev_v_z = (rev_v - rev_v.mean()) / rev_std

            # OLS: trend_v = alpha + beta * rev_v_z + eps
            X = np.column_stack([np.ones(n_valid), rev_v_z])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, trend_v, rcond=None)
                residuals = trend_v - X @ coeffs
            except np.linalg.LinAlgError:
                factor_values[:, t] = np.nan
                continue

            out = np.full(N, np.nan, dtype=np.float64)
            out[valid] = residuals
            factor_values[:, t] = out

        nan_ratio = np.isnan(factor_values).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"BP_MRQ_TREND NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=factor_values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python bp_mrq_trend.py)
# Uses full market with 2-year range + 3-year warmup for TREND.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("BP_MRQ_TREND factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute BP_MRQ_TREND factor")
    calculator = BP_MRQ_TREND()
    result = calculator.calculate(fd)

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range   : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # ---- Smoke-test assertions ----------------------------------------
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, \
        f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"NaN ratio    : {nan_ratio:.1%}")

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid    : {len(valid_cs)}")
        print(f"  mean       : {valid_cs.mean():.6f}")
        print(f"  std        : {valid_cs.std():.6f}")
        print(f"  min        : {valid_cs.min():.4f}")
        print(f"  max        : {valid_cs.max():.4f}")
        print(f"  median     : {np.median(valid_cs):.6f}")

    print(f"\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31", stock_codes=None)
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
