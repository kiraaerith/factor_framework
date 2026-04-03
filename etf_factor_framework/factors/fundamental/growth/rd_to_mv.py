"""
RD_TO_MV factor (R&D Expense to Market Cap Ratio, TTM)

Formula: RD_TTM / MarketCap
  - RD_TTM:    TTM rolling sum of q_ps_rade_c (single-quarter R&D expense, yuan)
  - MarketCap: daily mc from lixinger.fundamental (total market cap, yuan)

RD_TTM is computed from lixinger.financial_statements; TTM = rolling sum of 4 most
recently disclosed single-quarter values (ordered by report_date, not calendar date).
MarketCap is a daily field (no TTM needed), forward-filled within get_market_cap_panel().

Factor direction: positive (higher R&D-to-market-cap ratio means market underprices
                            R&D investment; similar to EP/BP value factors)
Factor category: growth - R&D investment

Unit alignment: both q_ps_rade_c and mc are in yuan, direct division is valid.

Boundary conditions:
  - MarketCap <= 0  -> NaN  (market cap should be strictly positive)
  - RD_TTM is NaN   -> NaN  (company did not disclose R&D; missing = info)
  - RD_TTM < 0      -> NaN  (data anomaly; R&D should not be negative)
  - RD_to_MV > 1   -> kept (extreme but rare; downstream Winsorize handles it)
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

FACTOR_NAME = "RD_TO_MV"
FACTOR_DIRECTION = 1  # positive: higher R&D-to-market-cap ratio is better


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class RD_TO_MV(FundamentalFactorCalculator):
    """
    R&D Expense to Market Cap Ratio (TTM).

    RD_TO_MV = RD_TTM / MarketCap
    where:
      RD_TTM    = rolling 4-quarter sum of q_ps_rade_c (R&D expense, single-quarter, yuan)
      MarketCap = daily total market cap mc from lixinger.fundamental (yuan)

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

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute RD_TO_MV daily panel (N stocks x T days).

        Returns:
            FactorData with values = RD_TTM / MarketCap (dimensionless ratio)
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        # --- 1. Compute RD TTM panel from financial_statements ---
        rd_v, rd_s, _ = self._compute_ttm_panel(
            fundamental_data, "q_ps_rade_c", trading_dates_arr
        )

        if rd_v.size == 0:
            raise ValueError("RD_TO_MV: q_ps_rade_c TTM panel is empty")

        # --- 2. Get daily market cap panel from fundamental ---
        mc_v, mc_s, _ = fundamental_data.get_market_cap_panel()

        if mc_v.size == 0:
            raise ValueError("RD_TO_MV: market cap panel is empty")

        # --- 3. Common symbols (mainboard only) ---
        common_syms = np.intersect1d(rd_s, mc_s)
        # Apply A-share mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in common_syms])
        common_syms = common_syms[mainboard_mask]

        if len(common_syms) == 0:
            raise ValueError("RD_TO_MV: no common mainboard symbols between RD and MarketCap panels")

        rd_aligned = rd_v[np.searchsorted(rd_s, common_syms)]   # (N, T)
        mc_aligned = mc_v[np.searchsorted(mc_s, common_syms)]   # (N, T)

        # --- 4. Compute ratio with boundary conditions ---
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                (mc_aligned > 0)
                & ~np.isnan(rd_aligned)
                & (rd_aligned >= 0),
                rd_aligned / mc_aligned,
                np.nan,
            )
        ratio = ratio.astype(np.float64)

        nan_ratio = np.isnan(ratio).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"RD_TO_MV NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=ratio,
            symbols=common_syms,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal helper: TTM panel from single-quarter field
    # ------------------------------------------------------------------

    def _compute_ttm_panel(
        self,
        fundamental_data: FundamentalData,
        field: str,
        target_dates: np.ndarray,
    ):
        """
        Build daily TTM panel by summing the 4 most recently disclosed single-quarter values.

        Ordered by report_date (disclosure date) to prevent future data leakage.
        Forward-fills TTM value from report_date until next disclosure.
        Max forward-fill: 180 trading days (~3 quarters); older values become NaN.

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

        if field not in raw.columns:
            return empty

        df = raw[["stock_code", "report_date", field]].dropna(subset=[field]).copy()
        if df.empty:
            return empty

        df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)
        df = df[df["report_date"] <= fundamental_data.end_date]
        if df.empty:
            return empty

        # Sort by (stock_code, report_date) ascending for rolling
        df = df.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

        # Rolling sum of 4 most recently disclosed quarters; require min 4
        df["ttm_val"] = (
            df.groupby("stock_code")[field]
            .transform(lambda x: x.rolling(4, min_periods=4).sum())
        )

        ttm_df = df[["stock_code", "report_date", "ttm_val"]].dropna(subset=["ttm_val"])
        if ttm_df.empty:
            return empty

        ttm_df = ttm_df.copy()
        ttm_df["report_date"] = pd.to_datetime(ttm_df["report_date"])

        # Map stock_code -> DuckDB symbol
        symbol_map = (
            raw[["stock_code", "symbol"]]
            .drop_duplicates("stock_code")
            .set_index("stock_code")["symbol"]
        )
        ttm_df["symbol"] = ttm_df["stock_code"].map(symbol_map)
        ttm_df = ttm_df.dropna(subset=["symbol"])
        if ttm_df.empty:
            return empty

        # Pivot -> forward-fill (max 180 days) -> reindex to trading dates
        pivot = ttm_df.pivot_table(
            index="symbol",
            columns="report_date",
            values="ttm_val",
            aggfunc="last",
        )

        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1, limit=180).reindex(columns=trading_dates)

        values = panel.values.astype(np.float64)

        sorted_syms = sorted(panel.index.tolist())
        sorted_idx = np.argsort(panel.index.tolist())
        values = values[sorted_idx]

        symbols_arr = np.array(sorted_syms)
        dates_arr = np.array(panel.columns.tolist(), dtype="datetime64[ns]")

        return values, symbols_arr, dates_arr


# ------------------------------------------------------------------
# Smoke test (run: python rd_to_mv.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("RD_TO_MV factor smoke test")
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

    print(f"\n[Step 2] Data validation: check required fields")
    v, s, d = fd.get_daily_panel("q_ps_rade_c")
    nr = np.isnan(v).mean()
    print(f"  q_ps_rade_c: shape={v.shape}, NaN={nr:.1%}")
    assert v.size > 0, "q_ps_rade_c returned empty array"

    mc_v, mc_s, mc_d = fd.get_market_cap_panel()
    mc_nr = np.isnan(mc_v).mean()
    print(f"  mc panel: shape={mc_v.shape}, NaN={mc_nr:.1%}")
    assert mc_v.size > 0, "mc panel returned empty array"

    print(f"\n[Step 3] Compute RD_TO_MV factor")
    calculator = RD_TO_MV()
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
        print(f"  mean: {valid.mean():.6f}")
        print(f"  median: {np.median(valid):.6f}")
        print(f"  min: {valid.min():.6f}")
        print(f"  max: {valid.max():.6f}")

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
