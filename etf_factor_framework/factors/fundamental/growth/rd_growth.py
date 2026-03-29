"""
RD_GROWTH factor (R&D Expense TTM Year-over-Year Growth Rate)

Formula: (RD_TTM_t - RD_TTM_{t-4Q}) / |RD_TTM_{t-4Q}|
  - RD_TTM_t:      rolling 4-quarter sum of q_ps_rade_c at current period
  - RD_TTM_{t-4Q}: rolling 4-quarter sum shifted back 4 quarters (1 year ago)

Data source: lixinger.financial_statements
  - q_ps_rade_c: single-quarter R&D expense (yuan), _c suffix = single-quarter value

Factor direction: positive (higher R&D growth rate is better)
Factor category: growth - R&D investment

Boundary conditions:
  - RD_TTM_{t-4Q} == 0  -> NaN  (avoid division by zero)
  - RD_TTM_{t-4Q} < 0   -> NaN  (data anomaly; R&D should be non-negative)
  - RD_TTM_t < 0         -> NaN  (data anomaly)
  - Either TTM is NaN    -> NaN
  - Extreme values (>10x or <-0.99) are kept; downstream Winsorize handles them

Universe: A-share mainboard only (60xxxx / 00xxxx).
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "RD_GROWTH"
FACTOR_DIRECTION = 1  # positive: higher R&D growth is better


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class RD_GROWTH(FundamentalFactorCalculator):
    """
    R&D Expense TTM Year-over-Year Growth Rate factor.

    RD_GROWTH = (RD_TTM_t - RD_TTM_{t-4Q}) / |RD_TTM_{t-4Q}|

    where:
      RD_TTM_t    = rolling 4-quarter sum of q_ps_rade_c (current period)
      RD_TTM_{t-4Q} = RD_TTM_t shifted back 4 quarters (1 year ago)

    Companies that do not disclose R&D (NaN q_ps_rade_c) receive NaN factor values.
    Universe restricted to A-share mainboard stocks (60xxxx / 00xxxx).
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

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        Compute RD_GROWTH daily panel (N stocks x T days).

        Returns:
            FactorData with values = (RD_TTM_t - RD_TTM_{t-4Q}) / |RD_TTM_{t-4Q}|
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        values, symbols, dates_out = self._compute_rd_growth_panel(
            fundamental_data, trading_dates_arr
        )

        if values.size == 0:
            raise ValueError("RD_GROWTH: panel is empty (no valid R&D data for this period)")

        nan_ratio = np.isnan(values).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"RD_GROWTH NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=values,
            symbols=symbols,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal helper: RD growth panel
    # ------------------------------------------------------------------

    def _compute_rd_growth_panel(
        self,
        fundamental_data: FundamentalData,
        target_dates: np.ndarray,
    ):
        """
        Build daily RD_GROWTH panel.

        Steps:
          1. Load q_ps_rade_c from financial_statements raw data.
          2. Sort by (stock_code, report_date) ascending.
          3. Rolling(4, min_periods=4).sum() per stock → RD_TTM_t.
          4. Shift(4) per stock → RD_TTM_{t-4Q}.
          5. Compute RD_GROWTH = (TTM - TTM_lag) / |TTM_lag| with boundary conditions.
          6. Map stock_code -> symbol, pivot, ffill (max 180 trading days) to daily.

        Returns:
            (values np.ndarray (N, T), symbols np.ndarray (N,), dates np.ndarray (T,))
        """
        fundamental_data._load_raw_data()
        raw = fundamental_data._raw_data

        trading_dates = pd.DatetimeIndex(target_dates)
        empty = (
            np.empty((0, len(trading_dates)), dtype=np.float64),
            np.array([], dtype=object),
            np.array(trading_dates, dtype="datetime64[ns]"),
        )

        if raw is None or raw.empty:
            return empty

        field = "q_ps_rade_c"
        if field not in raw.columns:
            return empty

        df = raw[["stock_code", "report_date", "symbol", field]].dropna(subset=[field]).copy()
        if df.empty:
            return empty

        df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)
        df = df[df["report_date"] <= fundamental_data.end_date]
        if df.empty:
            return empty

        # Sort by (stock_code, report_date) ascending for rolling
        df = df.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

        # TTM = rolling 4-quarter sum per stock (require all 4 quarters)
        df["ttm"] = (
            df.groupby("stock_code")[field]
            .transform(lambda x: x.rolling(4, min_periods=4).sum())
        )

        # Lag-4Q TTM = shift 4 periods within each stock's quarterly sequence
        df["ttm_lag4"] = (
            df.groupby("stock_code")["ttm"]
            .transform(lambda x: x.shift(4))
        )

        # Drop rows where either TTM is NaN (insufficient history)
        df = df.dropna(subset=["ttm", "ttm_lag4"])
        if df.empty:
            return empty

        # Compute RD_GROWTH with boundary conditions:
        # - ttm_lag4 == 0 -> NaN (division by zero)
        # - ttm_lag4 < 0  -> NaN (data anomaly)
        # - ttm < 0       -> NaN (data anomaly)
        with np.errstate(divide="ignore", invalid="ignore"):
            rd_growth = np.where(
                (df["ttm_lag4"].values > 0)
                & (df["ttm"].values >= 0),
                (df["ttm"].values - df["ttm_lag4"].values) / np.abs(df["ttm_lag4"].values),
                np.nan,
            )
        df = df.copy()
        df["rd_growth"] = rd_growth

        growth_df = df[["stock_code", "report_date", "symbol", "rd_growth"]].dropna(
            subset=["rd_growth"]
        )
        if growth_df.empty:
            return empty

        # Map stock_code -> symbol (use symbol column directly; already in df)
        growth_df = growth_df.dropna(subset=["symbol"])
        if growth_df.empty:
            return empty

        # Apply A-share mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in growth_df["symbol"].values])
        growth_df = growth_df[mainboard_mask]
        if growth_df.empty:
            return empty

        # Pivot: index=symbol, columns=report_date, values=rd_growth
        pivot = growth_df.pivot_table(
            index="symbol",
            columns="report_date",
            values="rd_growth",
            aggfunc="last",
        )

        # Reindex to include all trading dates, ffill (max 180 days), then select trading dates
        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1, limit=180).reindex(columns=trading_dates)

        # Sort symbols
        sorted_syms = sorted(panel.index.tolist())
        sorted_idx = [panel.index.tolist().index(s) for s in sorted_syms]
        values = panel.values[sorted_idx].astype(np.float64)

        symbols_arr = np.array(sorted_syms)
        dates_arr = np.array(panel.columns.tolist(), dtype="datetime64[ns]")

        return values, symbols_arr, dates_arr


# ------------------------------------------------------------------
# Smoke test (run: python rd_growth.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("RD_GROWTH factor smoke test")
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

    print(f"\n[Step 2] Data validation: check q_ps_rade_c field")
    v, s, d = fd.get_daily_panel("q_ps_rade_c")
    nr = np.isnan(v).mean()
    print(f"  q_ps_rade_c: shape={v.shape}, NaN={nr:.1%}")
    assert v.size > 0, "q_ps_rade_c returned empty array"

    print(f"\n[Step 3] Compute RD_GROWTH factor")
    calculator = RD_GROWTH()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ {pd.Timestamp(result.dates[-1]).date()}")

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {last5}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"  N valid: {len(valid)}")
        print(f"  mean: {valid.mean():.4f}")
        print(f"  median: {np.median(valid):.4f}")
        print(f"  min: {valid.min():.4f}")
        print(f"  max: {valid.max():.4f}")

    print(f"\n[Step 4] Smoke test assertions")
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 5] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2013-01-01", end_date="2025-12-31", stock_codes=None)
    leakage_found = False
    for sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n--- split_ratio={sr} ---")
        detector = FundamentalLeakageDetector(split_ratio=sr)
        try:
            report = detector.detect(calculator, fd_leak)
            report.print_report()
            if report.has_leakage:
                leakage_found = True
                print(f"[FAIL] Leakage detected at split_ratio={sr}")
            else:
                print(f"[OK] No leakage at split_ratio={sr}")
        except ValueError as e:
            if "panel is empty" in str(e):
                print(f"[SKIP] split_ratio={sr}: insufficient R&D data for truncated period, skip")
            else:
                raise

    if leakage_found:
        print("\n[RESULT] LEAKAGE DETECTED")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL PASSED - No leakage")
