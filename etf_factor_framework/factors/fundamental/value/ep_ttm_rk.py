"""
EP_TTM_RK factor (EP TTM Time-Series Rank, Purified from 6-Month Reversal)

Formula:
  1. EP_TTM = 1 / pe_ttm  (daily, from lixinger.fundamental)
  2. RK_t = rolling 126-day ascending rank of EP_TTM per stock / count_valid
             (time-series rank, per-stock, window=126 trading days, min_periods=63)
  3. REV_6M_t = close_t / close_{t-126} - 1  (from juejin ohlcv_adjusted, 126 trading days)
  4. EP_TTM_RK = residual of cross-section OLS: RK_t ~ z-score(REV_6M_t)

Data fields:
  - pe_ttm  : lixinger.fundamental (daily valuation)
  - close   : juejin ohlcv_adjusted (daily adjusted close price)

Factor direction: positive (higher = relatively cheaper valuation historically, net of reversal)
Factor category: value - time-series rank
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

FACTOR_NAME = "EP_TTM_RK"
FACTOR_DIRECTION = 1    # positive: higher rank = cheaper relative to own history = better
MIN_ABS_PE = 1e-6       # near-zero PE threshold for NaN assignment
RK_WINDOW = 126         # rolling window in trading days (~6 months)
RK_MIN_PERIODS = 63     # minimum non-NaN observations for rank (half window)
MIN_VALID_STOCKS = 100  # minimum cross-section size for reliable OLS
PRICE_LOOKBACK_DAYS = 220  # calendar days before start_date for close price loading


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class EP_TTM_RK(FundamentalFactorCalculator):
    """
    EP TTM Time-Series Rank factor, purified from 6-month reversal.

    Steps:
      1. EP_TTM = 1/pe_ttm (lixinger.fundamental, daily)
      2. RK = 126-day rolling time-series rank of EP_TTM, per stock
      3. REV_6M = 126-trading-day cumulative return (juejin ohlcv_adjusted)
      4. EP_TTM_RK = residual of cross-section OLS: RK ~ z-score(REV_6M), per day

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
        Calculate EP_TTM_RK daily panel.

        Returns:
            FactorData: N stocks x T days, values = OLS residuals (float64)
        """
        # ------------------------------------------------------------------
        # Step 1: Load pe_ttm, compute EP_TTM, apply mainboard filter
        # ------------------------------------------------------------------
        pe_values, symbols, dates = fundamental_data.get_valuation_panel("pe_ttm")

        mb_mask = np.array([_is_mainboard(s) for s in symbols])
        pe_values = pe_values[mb_mask]
        symbols = symbols[mb_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ep_values = np.where(
                np.abs(pe_values) < MIN_ABS_PE,
                np.nan,
                1.0 / pe_values,
            )
        ep_values = np.where(np.isinf(ep_values), np.nan, ep_values).astype(np.float64)

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
        # Step 2: Rolling 126-day time-series rank per stock
        # DataFrame shape: (T, N) so rolling().rank() works column-wise
        # ------------------------------------------------------------------
        ep_df = pd.DataFrame(ep_values.T, index=trading_dates, columns=symbols)
        rk_df = ep_df.rolling(window=RK_WINDOW, min_periods=RK_MIN_PERIODS).rank(pct=True)
        rk_values = rk_df.values.T.astype(np.float64)  # (N, T)

        # ------------------------------------------------------------------
        # Step 3: Load close prices from DuckDB, compute REV_6M
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

        rev6m_values = np.full((N, T), np.nan, dtype=np.float64)

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

            # REV_6M = close_t / close_{t-126} - 1  (position-based shift)
            close_shifted = close_panel.shift(RK_WINDOW, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rev6m_panel = close_panel / close_shifted - 1.0

            # Extract values at factor trading_dates only
            # Reindex panel to symbols (mainboard EP panel) and trading_dates
            rev6m_aligned = (
                rev6m_panel
                .reindex(index=symbols)
                .reindex(columns=trading_dates)
            )
            rev6m_values = rev6m_aligned.values.astype(np.float64)

        # ------------------------------------------------------------------
        # Step 4: Cross-section OLS per day: RK ~ z-score(REV_6M), residuals
        # ------------------------------------------------------------------
        factor_values = rk_values.copy()  # fallback: use RK directly if OLS fails

        for t in range(T):
            rk_t = rk_values[:, t]
            rev_t = rev6m_values[:, t]

            valid = ~np.isnan(rk_t) & ~np.isnan(rev_t)
            n_valid = int(valid.sum())

            if n_valid < MIN_VALID_STOCKS:
                # Not enough stocks; keep rk_t as-is
                continue

            rk_v = rk_t[valid]
            rev_v = rev_t[valid]

            # Z-score standardize REV_6M
            rev_std = rev_v.std()
            if rev_std < 1e-10:
                continue
            rev_v_z = (rev_v - rev_v.mean()) / rev_std

            # OLS: rk_v = alpha + beta * rev_v_z + eps
            X = np.column_stack([np.ones(n_valid), rev_v_z])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, rk_v, rcond=None)
                residuals = rk_v - X @ coeffs
            except np.linalg.LinAlgError:
                continue

            # Assign residuals; invalid positions remain NaN (already set from rk_t which was NaN)
            out = np.full(N, np.nan, dtype=np.float64)
            out[valid] = residuals
            # Stocks with valid RK but no REV_6M: keep their RK value (already in factor_values)
            # Stocks with valid both: overwrite with residual
            factor_values[:, t] = out

        nan_ratio = np.isnan(factor_values).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"EP_TTM_RK NaN ratio is high: {nan_ratio:.1%}, please check data"
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
# Smoke test (run: python ep_ttm_rk.py)
# Uses full market with 2-year range to ensure OLS has enough cross-section.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("EP_TTM_RK factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute EP_TTM_RK factor")
    calculator = EP_TTM_RK()
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
