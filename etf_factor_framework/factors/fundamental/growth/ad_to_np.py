"""
AD_TO_NP factor (Selling Expense to Net Profit Ratio, TTM)

Formula: SE_TTM / NetProfit_TTM
  - SE_TTM:         TTM rolling sum of q_ps_se_c         (single-quarter selling expense)
  - NetProfit_TTM:  TTM rolling sum of q_ps_npatoshopc_c (single-quarter net profit attr. to parent)

Both fields from lixinger.financial_statements; TTM = rolling sum of 4 most recently
disclosed single-quarter values (ordered by report_date, not calendar date).

Note: lixinger has no separate advertising expense field. Selling expense (q_ps_se_c) is
used as the best proxy for advertising/promotion costs. Factor name is AD_TO_NP but
actual numerator is selling expense TTM.

Factor direction: positive (higher selling reinvestment intensity relative to profit predicts
                            stronger brand building and long-term competitive moat)
Factor category: growth - advertising/selling investment

Boundary conditions:
  - NetProfit_TTM <= 0              -> NaN  (loss firms / zero profit: ratio direction invalid)
  - SE_TTM is NaN                   -> NaN  (company did not disclose selling expense)
  - SE_TTM < 0                      -> NaN  (data anomaly; selling expense should not be negative)
  - AD_to_NP > 5 (extreme outlier)  -> kept (downstream Winsorize handles it)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "AD_TO_NP"
FACTOR_DIRECTION = 1  # positive: higher selling reinvestment rate is better


class AD_TO_NP(FundamentalFactorCalculator):
    """
    Selling Expense to Net Profit Ratio (TTM), used as proxy for AD_TO_NP.

    AD_TO_NP = SE_TTM / NetProfit_TTM
    where:
      SE_TTM        = rolling 4-quarter sum of q_ps_se_c          (selling expense, single-quarter)
      NetProfit_TTM = rolling 4-quarter sum of q_ps_npatoshopc_c  (net profit attr. to parent,
                                                                    single-quarter)

    Companies with non-positive or zero net profit (TTM) receive NaN (ratio direction invalid).
    Companies that do not disclose selling expense (NaN q_ps_se_c) receive NaN factor values.
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
        Compute AD_TO_NP daily panel (N stocks x T days).

        Returns:
            FactorData with values = SE_TTM / NetProfit_TTM (dimensionless ratio)
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        # --- 1. Compute TTM panels ---
        se_v, se_s, se_d = self._compute_ttm_panel(
            fundamental_data, "q_ps_se_c", trading_dates_arr
        )
        np_v, np_s, np_d = self._compute_ttm_panel(
            fundamental_data, "q_ps_npatoshopc_c", trading_dates_arr
        )

        if se_v.size == 0:
            raise ValueError("AD_TO_NP: q_ps_se_c TTM panel is empty")
        if np_v.size == 0:
            raise ValueError("AD_TO_NP: q_ps_npatoshopc_c TTM panel is empty")

        # --- 2. Common symbols ---
        common_syms = np.intersect1d(se_s, np_s)
        if len(common_syms) == 0:
            raise ValueError("AD_TO_NP: no common symbols between SE and NetProfit panels")

        N = len(common_syms)
        T = len(trading_dates_arr)

        se_si = np.searchsorted(se_s, common_syms)
        np_si = np.searchsorted(np_s, common_syms)

        se_aligned = se_v[se_si]   # (N, T)
        np_aligned = np_v[np_si]   # (N, T)

        # --- 3. Compute ratio with boundary conditions ---
        # NetProfit_TTM must be strictly positive (loss/zero: ratio direction invalid)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                (np_aligned > 0)
                & ~np.isnan(se_aligned)
                & (se_aligned >= 0),
                se_aligned / np_aligned,
                np.nan,
            )
        ratio = ratio.astype(np.float64)

        nan_ratio = np.isnan(ratio).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"AD_TO_NP NaN ratio is high: {nan_ratio:.1%}, please check data"
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
# Smoke test (run: python ad_to_np.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("AD_TO_NP factor smoke test")
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
    for field in ["q_ps_se_c", "q_ps_npatoshopc_c"]:
        v, s, d = fd.get_daily_panel(field)
        nr = np.isnan(v).mean()
        print(f"  {field}: shape={v.shape}, NaN={nr:.1%}")
        assert v.size > 0, f"{field} returned empty array"

    print(f"\n[Step 3] Compute AD_TO_NP factor")
    calculator = AD_TO_NP()
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
