"""
ROBUST_GROWTH factor (Robust Growth Rate)

Computes: g_robust = (x_t - x_{t-T}) / std(x_{t-T..t})
where std is the population standard deviation of the T+1 quarterly data points.

Measures growth per unit of volatility within a window, rewarding consistent,
directional growth while penalizing erratic patterns.

Two time windows (T in quarters):
  - T=4  (1-year: 5 data points, current + 4 prior quarters)
  - T=20 (5-year: 21 data points, current + 20 prior quarters)

Applied to 4 raw quarterly financial metrics:
  - q_ps_toi_c        : total operating income (single-quarter)
  - q_ps_npatoshopc_c : attributable net profit (single-quarter)
  - q_ps_op_c         : operating profit (single-quarter)
  - q_cfs_ncffoa_c    : operating cash flow net (single-quarter)

Synthesis:
  1. Cross-sectional Winsorize raw field values at each fiscal date (1%-99%).
  2. Compute g_robust per symbol for each (field, T) combination (PIT-safe).
  3. Forward-fill quarterly g_robust to daily panel via report_date.
  4. Winsorize g_robust daily panel cross-sectionally (1%-99%).
  5. Z-score each daily panel cross-sectionally.
  6. Equal-weight average across all (field, T) combinations (max 8).
  7. Cross-sectional rank normalize to [0, 1].
  8. Filter to A-share mainboard stocks (60xxxx / 00xxxx).

Data source : lixinger.financial_statements (fs_type='q')
Factor direction : positive (higher robust growth is better)
Factor category  : growth - robust growth method

Notes:
  - PIT-safe: for each quarterly row i, only uses prior rows with
    report_date <= current report_date when building the T+1 window.
  - Deduplication: for same (symbol, fiscal date), earliest report_date is kept.
  - std=0 (no variation) yields NaN for that data point.
  - Output is restricted to A-share mainboard stocks (60xxxx / 00xxxx).
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "ROBUST_GROWTH"
FACTOR_DIRECTION = 1  # positive: higher robust growth is better

ROBUST_FIELDS = [
    "q_ps_toi_c",
    "q_ps_npatoshopc_c",
    "q_ps_op_c",
    "q_cfs_ncffoa_c",
]

ROBUST_WINDOWS = [4, 20]

WINSOR_LO = 1.0   # percentile
WINSOR_HI = 99.0  # percentile


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _cross_section_winsorize_by_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Cross-sectional Winsorize col at each fiscal 'date' (1%-99%).
    Returns a copy with col replaced by clipped values.
    """
    df = df.copy()

    def _winsor(x):
        if len(x) < 10:
            return x
        lo = np.nanpercentile(x.values, WINSOR_LO)
        hi = np.nanpercentile(x.values, WINSOR_HI)
        return x.clip(lower=lo, upper=hi)

    df[col] = df.groupby("date")[col].transform(_winsor)
    return df


def _build_quarterly_panel(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Extract quarterly records for field, deduplicate, and sort.

    Deduplication: same (symbol, fiscal date) -> keep earliest report_date.
    Sanity filter: fiscal date (date) must not exceed its report_date.
    """
    needed = ["symbol", "date", "report_date", field]
    df = raw[needed].dropna(subset=[field]).copy()

    # Sanity: fiscal period end must not be after disclosure date
    df = df[df["date"] <= df["report_date"]]

    # Deduplicate: same (symbol, fiscal date) -> keep earliest report_date
    df = df.sort_values(["symbol", "date", "report_date"])
    df = df.groupby(["symbol", "date"], as_index=False).first()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


def _compute_robust_growth_series(
    vals_arr: np.ndarray,
    rd_arr: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Compute ROBUST_GROWTH for one symbol's quarterly series.

    Args:
        vals_arr : float64 array (n,) - quarterly field values sorted by fiscal date
        rd_arr   : datetime64[ns] array (n,) - corresponding report_dates
        T        : int - lookback window in quarters

    Returns:
        float64 array (n,) of g_robust values

    PIT safety:
        For row i, only quarters with report_date <= rd_arr[i] are PIT-valid.
        Takes the last T+1 such quarters as the rolling window.
        This avoids positional-shift bugs when late-disclosed quarters are
        excluded in truncated datasets.
    """
    n = len(vals_arr)
    results = np.full(n, np.nan)

    for i in range(n):
        cur_rd = rd_arr[i]

        # Collect indices of all PIT-valid quarters up to i
        pit_ok = rd_arr[:i + 1] <= cur_rd
        pit_indices = np.where(pit_ok)[0]

        # Need T+1 data points
        if len(pit_indices) < T + 1:
            continue

        window_indices = pit_indices[-(T + 1):]
        x = vals_arr[window_indices].astype(float)

        # Skip if any NaN in window
        if np.any(np.isnan(x)):
            continue

        x_t = x[-1]   # current (most recent) value
        x_tT = x[0]   # T quarters ago value
        std_val = np.std(x, ddof=0)  # population std of T+1 points

        if std_val == 0.0:
            continue  # all values identical, no variation

        results[i] = (x_t - x_tT) / std_val

    return results


def _build_daily_panel_from_quarterly(
    qdf: pd.DataFrame,
    col: str,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Forward-fill quarterly col values to a daily panel (symbol x trading_dates)
    using report_date as the signal availability date.
    """
    df = qdf[["symbol", "report_date", col]].dropna(subset=[col]).copy()
    df = df[df["report_date"] <= end_date]

    if df.empty:
        return pd.DataFrame(index=[], columns=trading_dates)

    pivot = df.pivot_table(
        index="symbol", columns="report_date", values=col, aggfunc="last"
    )
    all_dates = pivot.columns.union(trading_dates).sort_values()
    pivot = pivot.reindex(columns=all_dates)
    panel = pivot.ffill(axis=1).reindex(columns=trading_dates)
    return panel


def _cross_section_winsorize_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize each date column cross-sectionally (1%-99%)."""
    arr = panel_df.values.astype(np.float64)
    result = arr.copy()
    for t in range(arr.shape[1]):
        col = arr[:, t]
        valid_mask = ~np.isnan(col)
        n_valid = valid_mask.sum()
        if n_valid < 10:
            continue
        lo = np.percentile(col[valid_mask], WINSOR_LO)
        hi = np.percentile(col[valid_mask], WINSOR_HI)
        result[valid_mask, t] = np.clip(col[valid_mask], lo, hi)
    return pd.DataFrame(result, index=panel_df.index, columns=panel_df.columns)


def _cross_section_zscore(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each date column cross-sectionally."""
    arr = panel_df.values.astype(np.float64)
    result = np.full_like(arr, np.nan)
    for t in range(arr.shape[1]):
        col = arr[:, t]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() < 2:
            continue
        mu = col[valid_mask].mean()
        sigma = col[valid_mask].std()
        if sigma < 1e-8:
            continue
        result[valid_mask, t] = (col[valid_mask] - mu) / sigma
    return pd.DataFrame(result, index=panel_df.index, columns=panel_df.columns)


def _cross_section_rank_norm(arr: np.ndarray) -> np.ndarray:
    """Rank-normalize each date column to [0, 1]."""
    result = np.full_like(arr, np.nan)
    for t in range(arr.shape[1]):
        col = arr[:, t]
        valid_mask = ~np.isnan(col)
        n_valid = valid_mask.sum()
        if n_valid < 2:
            continue
        ranks = pd.Series(col[valid_mask]).rank(method="average").values
        result[valid_mask, t] = (ranks - 1) / (n_valid - 1)
    return result


class ROBUST_GROWTH(FundamentalFactorCalculator):
    """
    Robust Growth Rate factor.

    Measures growth per unit of volatility:
        g_robust = (x_t - x_{t-T}) / std(x_{t-T..t})

    Applied to 4 fundamental metrics over two time windows (T=4, T=20 quarters),
    synthesized via cross-sectional z-score + equal-weight average + rank normalization.
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
            "fields": ROBUST_FIELDS,
            "windows": ROBUST_WINDOWS,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute ROBUST_GROWTH daily panel (N stocks x T_days).

        Steps:
          1. Load raw quarterly data for all 4 fields.
          2. Per field: cross-sectional Winsorize raw values by fiscal date.
          3. Per (field, T): compute g_robust per symbol (PIT-safe rolling window).
          4. Forward-fill quarterly g_robust to daily panel via report_date.
          5. Winsorize g_robust panel cross-sectionally at each date.
          6. Z-score each daily panel cross-sectionally.
          7. Equal-weight average across all (field, T) pairs.
          8. Cross-sectional rank normalize to [0, 1].
          9. Filter to A-share mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        end_date = fundamental_data.end_date

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("ROBUST_GROWTH: no raw fundamental data available")

        raw = fundamental_data._raw_data
        zscored_panels = []

        for field in ROBUST_FIELDS:
            # Step 1: Extract and deduplicate quarterly data
            qdf = _build_quarterly_panel(raw, field)
            if qdf.empty:
                continue

            for T in ROBUST_WINDOWS:
                rg_col = f"rg_{field}_T{T}"
                rg_vals = np.full(len(qdf), np.nan)

                # Step 3: Compute g_robust per symbol
                for symbol, grp in qdf.groupby("symbol", sort=False):
                    grp_sorted = grp.sort_values("date")
                    idx = grp_sorted.index
                    vals = grp_sorted[field].values.astype(float)
                    rd_arr = grp_sorted["report_date"].values.astype("datetime64[ns]")
                    rg_arr = _compute_robust_growth_series(vals, rd_arr, T)
                    rg_vals[idx] = rg_arr

                qdf[rg_col] = rg_vals

                # Step 4: Forward-fill to daily panel via report_date
                daily_panel = _build_daily_panel_from_quarterly(
                    qdf, rg_col, trading_dates, end_date
                )
                if daily_panel.empty or daily_panel.shape[0] == 0:
                    continue

                # Step 5: Winsorize g_robust cross-sectionally
                daily_panel = _cross_section_winsorize_panel(daily_panel)

                # Step 6: Z-score cross-sectionally
                zs_panel = _cross_section_zscore(daily_panel)
                zscored_panels.append(zs_panel)

        if not zscored_panels:
            raise ValueError("ROBUST_GROWTH: no valid sub-panels computed")

        # Step 7: Align to union symbols and equal-weight average
        all_symbols = sorted(set().union(*[set(p.index.tolist()) for p in zscored_panels]))

        aligned = [
            p.reindex(index=all_symbols, columns=trading_dates)
            for p in zscored_panels
        ]
        stacked = np.stack([a.values for a in aligned], axis=0)  # (K, N, T_days)
        valid_count = np.sum(~np.isnan(stacked), axis=0)
        composite = np.nanmean(stacked, axis=0)
        composite[valid_count == 0] = np.nan

        # Step 8: Cross-sectional rank normalization
        composite = _cross_section_rank_norm(composite)

        # Step 9: Mainboard filter
        symbols_arr = np.array(all_symbols)
        mb_mask = np.array([_is_mainboard(s) for s in symbols_arr])
        composite = composite[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        if len(symbols_arr) == 0:
            raise ValueError("ROBUST_GROWTH: no mainboard stocks after filtering")

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(f"ROBUST_GROWTH NaN ratio is high: {nan_ratio:.1%}")

        dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        return FactorData(
            values=composite.astype(np.float64),
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python robust_growth.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("ROBUST_GROWTH factor smoke test")
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

    print(f"\n[Step 2] Compute ROBUST_GROWTH factor")
    calculator = ROBUST_GROWTH()
    result = calculator.calculate(fd)

    print(f"\nFactor shape   : {result.shape}")
    print(f"Symbols        : {result.symbols}")
    print(f"Date range     : {pd.Timestamp(result.dates[0]).date()} ~ {pd.Timestamp(result.dates[-1]).date()}")

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio      : {nan_ratio:.1%}")

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {last5}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"  N valid : {len(valid)}")
        print(f"  mean    : {valid.mean():.4f}")
        print(f"  median  : {np.median(valid):.4f}")
        print(f"  min     : {valid.min():.4f}")
        print(f"  max     : {valid.max():.4f}")
    else:
        print("  No valid values in last cross-section")

    # --- Section 4.1 Smoke Test Assertions ---
    print(f"\n[Step 3] Smoke test assertions")

    assert result.values.ndim == 2, "values must be 2-D"
    print(f"  [PASS] shape is 2-D: {result.values.shape}")

    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"
    print(f"  [PASS] dtype = float64")

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"  [PASS] NaN ratio < 80%: {nan_ratio:.1%}")

    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"
    print(f"  [PASS] No inf values")

    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"
    print(f"  [PASS] Idempotent")

    print(f"\n[PASS] All smoke test assertions passed!")
    print(f"[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 4] Leakage detection (5 split ratios)")
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
