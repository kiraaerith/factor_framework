"""
BP_MRQ_ZS factor (BP_MRQ 2-Year Time-Series Z-Score, Purified from 2-Year Reversal)

Formula:
  1. BP_MRQ = 1 / pb  (daily, from lixinger.fundamental)
     - pb < 0 => BP_MRQ < 0 (kept as-is)
     - |pb| < 1e-6 or NaN/inf => NaN
  2. ZS_t = (BP_MRQ_t - mean(BP_MRQ, window=504)) / std(BP_MRQ, window=504)
            (rolling 504-day time-series Z-Score per stock,
             min_periods=252, i.e. at least 1 year of data required;
             if std < 1e-9 => NaN to avoid division by near-zero)
  3. REV_2Y_t = close_t / close_{t-504} - 1  (from juejin ohlcv_adjusted, 504 trading days)
  4. BP_MRQ_ZS = residual of cross-section OLS: ZS_t ~ z-score(REV_2Y_t)

Data fields:
  - pb    : lixinger.fundamental (daily valuation)
  - close : juejin ohlcv_adjusted (daily adjusted close price)

Factor direction: positive (higher = BP currently high vs own history, net of reversal)
Factor category: value - time-series z-score
"""

import os
import re
import sys
import warnings

import sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "BP_MRQ_ZS"
FACTOR_DIRECTION = 1       # positive: higher Z-Score = BP high vs own history = cheaper
MIN_ABS_PB = 1e-6          # near-zero PB threshold for NaN assignment
ZS_WINDOW = 504            # rolling window in trading days (~2 years)
ZS_MIN_PERIODS = 252       # minimum non-NaN observations for Z-Score (~1 year)
ZS_MIN_STD = 1e-9          # minimum std to avoid near-zero division
MIN_VALID_STOCKS = 100     # minimum cross-section size for reliable OLS
PRICE_LOOKBACK_DAYS = 800  # calendar days before start_date for close price loading


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class BP_MRQ_ZS(FundamentalFactorCalculator):
    """
    BP_MRQ 2-Year Time-Series Z-Score factor, purified from 2-year reversal.

    Steps:
      1. BP_MRQ = 1/pb (lixinger.fundamental, daily)
      2. ZS = 504-day rolling time-series Z-Score of BP_MRQ, per stock
              (mean-centered and std-scaled, min_periods=252)
      3. REV_2Y = 504-trading-day cumulative return (juejin ohlcv_adjusted)
      4. BP_MRQ_ZS = residual of cross-section OLS: ZS ~ z-score(REV_2Y), per day

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
        Calculate BP_MRQ_ZS daily panel.

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
        # Step 2: Rolling 504-day time-series Z-Score per stock
        # DataFrame shape: (T, N) so rolling().mean()/std() works column-wise
        # ------------------------------------------------------------------
        bp_df = pd.DataFrame(bp_values.T, index=trading_dates, columns=symbols)

        roll = bp_df.rolling(window=ZS_WINDOW, min_periods=ZS_MIN_PERIODS)
        roll_mean = roll.mean()
        roll_std = roll.std(ddof=1)

        # Avoid division by near-zero std
        roll_std_safe = roll_std.where(roll_std >= ZS_MIN_STD, other=np.nan)

        zs_df = (bp_df - roll_mean) / roll_std_safe
        zs_values = zs_df.values.T.astype(np.float64)  # (N, T)

        # ------------------------------------------------------------------
        # Step 3: Load close prices from DuckDB, compute REV_2Y
        # ------------------------------------------------------------------
        extended_start = (fundamental_data.start_date - pd.DateOffset(days=PRICE_LOOKBACK_DAYS)).date()
        load_end = fundamental_data.end_date.date()

        tushare_db = fundamental_data._tushare_db
        ext_start_str = pd.Timestamp(extended_start).strftime("%Y%m%d")
        ext_end_str = pd.Timestamp(load_end).strftime("%Y%m%d")

        conn = sqlite3.connect(tushare_db)

        # Get extended trading calendar from tushare trade_cal
        ext_dates_df = pd.read_sql_query(
            f"""
            SELECT cal_date AS trade_date FROM trade_cal
            WHERE exchange = 'SSE' AND is_open = 1
              AND cal_date BETWEEN '{ext_start_str}' AND '{ext_end_str}'
            ORDER BY cal_date
            """, conn
        )
        ext_trading_dates = pd.DatetimeIndex(pd.to_datetime(ext_dates_df["trade_date"]))

        # Load close prices for all mainboard stocks from tushare daily_hfq
        price_df = pd.read_sql_query(
            f"""
            SELECT ts_code, trade_date, close
            FROM daily_hfq
            WHERE (ts_code LIKE '60%' OR ts_code LIKE '00%')
              AND trade_date BETWEEN '{ext_start_str}' AND '{ext_end_str}'
              AND close IS NOT NULL
            """, conn
        )
        conn.close()

        # Convert tushare code to juejin format
        def _ts_to_jq(ts_code):
            code, market = ts_code.split(".")
            return f"{'SHSE' if market == 'SH' else 'SZSE'}.{code}"

        price_df["symbol"] = price_df["ts_code"].apply(_ts_to_jq)
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
        print(f"  - Price data: {len(price_df)} rows, {price_df['symbol'].nunique()} stocks (tushare)")

        rev2y_values = np.full((N, T), np.nan, dtype=np.float64)

        if not price_df.empty:
            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

            # Pivot: (stocks, extended_trading_dates)
            close_panel = price_df.pivot_table(
                index="symbol", columns="trade_date", values="close", aggfunc="last"
            )
            # Reindex to full extended trading calendar
            close_panel = close_panel.reindex(columns=ext_trading_dates)
            # Forward-fill for suspended stocks
            close_panel = close_panel.ffill(axis=1)

            # REV_2Y = close_t / close_{t-504} - 1  (position-based shift)
            close_shifted = close_panel.shift(ZS_WINDOW, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rev2y_panel = close_panel / close_shifted - 1.0

            # Extract values at factor trading_dates only
            rev2y_aligned = (
                rev2y_panel
                .reindex(index=symbols)
                .reindex(columns=trading_dates)
            )
            rev2y_values = rev2y_aligned.values.astype(np.float64)

        # ------------------------------------------------------------------
        # Step 4: Cross-section OLS per day: ZS ~ z-score(REV_2Y), residuals
        # ------------------------------------------------------------------
        factor_values = zs_values.copy()  # fallback: use ZS directly if OLS fails

        for t in range(T):
            zs_t = zs_values[:, t]
            rev_t = rev2y_values[:, t]

            valid = ~np.isnan(zs_t) & ~np.isnan(rev_t)
            n_valid = int(valid.sum())

            if n_valid < MIN_VALID_STOCKS:
                # Not enough stocks; keep zs_t as-is
                continue

            zs_v = zs_t[valid]
            rev_v = rev_t[valid]

            # Z-score standardize REV_2Y
            rev_std = rev_v.std()
            if rev_std < 1e-10:
                continue
            rev_v_z = (rev_v - rev_v.mean()) / rev_std

            # OLS: zs_v = alpha + beta * rev_v_z + eps
            X = np.column_stack([np.ones(n_valid), rev_v_z])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, zs_v, rcond=None)
                residuals = zs_v - X @ coeffs
            except np.linalg.LinAlgError:
                continue

            # Assign residuals; stocks with valid both: overwrite with residual
            out = np.full(N, np.nan, dtype=np.float64)
            out[valid] = residuals
            factor_values[:, t] = out

        nan_ratio = np.isnan(factor_values).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"BP_MRQ_ZS NaN ratio is high: {nan_ratio:.1%}, please check data"
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
# Smoke test (run: python bp_mrq_zs.py)
# Uses full market with 2-year range to ensure OLS has enough cross-section.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("BP_MRQ_ZS factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute BP_MRQ_ZS factor")
    calculator = BP_MRQ_ZS()
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
