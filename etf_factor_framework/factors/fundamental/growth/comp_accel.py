"""
COMP_ACCEL factor (Composite Growth Acceleration via Linear Regression)

For each _y2y growth rate field, fits a linear trend over the last T quarterly
values and computes acc_comp = b / mean(g), where b is the OLS slope.
This measures the trend-based acceleration of growth rates (second-order growth).

Applied to 4 YoY growth rate fields:
  - q_ps_toi_c_y2y       : total operating income YoY growth
  - q_ps_npatoshopc_c_y2y: attributable net profit YoY growth
  - q_ps_op_c_y2y        : operating profit YoY growth
  - q_cfs_ncffoa_c_y2y   : operating cash flow net YoY growth

Two time windows (T in quarters):
  - T=4  (1-year window)
  - T=8  (2-year window, default per spec)

Synthesis:
  1. Winsorize g to [-5, 5] before computing COMP_ACCEL.
  2. If |mean_g| < 1e-6, set COMP_ACCEL to NaN (denominator too small).
  3. If valid periods < min_periods (T/2), set to NaN.
  4. For each (field, T) pair, z-score cross-sectionally at each date.
  5. Equal-weight average across all (field, T) combinations (max 8).
  6. Cross-sectional rank normalization to [0, 1].

Data source : lixinger.financial_statements (fs_type='q')
Factor direction : positive (higher composite acceleration = stronger momentum)
Factor category  : growth - acceleration method

Notes:
  - All data is PIT-safe: records at report_date j <= report_date i are used.
  - Output is restricted to A-share mainboard stocks (60xxxx / 00xxxx).
  - T-quarter lookback is positional (sorted by fiscal_date per symbol).
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

FACTOR_NAME = "COMP_ACCEL"
FACTOR_DIRECTION = 1  # positive: higher composite acceleration is better

# Fields to compute COMP_ACCEL on
COMP_ACCEL_FIELDS = [
    "q_ps_toi_c_y2y",
    "q_ps_npatoshopc_c_y2y",
    "q_ps_op_c_y2y",
    "q_cfs_ncffoa_c_y2y",
]

# Time windows (in quarters)
COMP_ACCEL_WINDOWS = [4, 8]

# Winsorization bounds for g values
G_WINSOR_LO = -5.0
G_WINSOR_HI = 5.0

# Minimum absolute mean_g to avoid division blow-up
MIN_ABS_MEAN_G = 1e-6


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _compute_comp_accel_quarterly(
    g_vals: np.ndarray,
    T: int,
    report_dates: np.ndarray,
) -> np.ndarray:
    """
    Compute COMP_ACCEL for one symbol's quarterly g series.

    For each record i, collects the last T PIT-safe quarterly g values
    (report_date[j] <= report_date[i]), fits an OLS line, and returns
    acc_comp = b / mean(g).

    Records must be sorted by fiscal_date ascending.

    Args:
        g_vals       : float64 array (n,) of winsorized YoY growth rates
        T            : int, lookback window in quarters
        report_dates : datetime64[ns] array (n,) of publication dates

    Returns:
        float64 array (n,) of COMP_ACCEL values
    """
    n = len(g_vals)
    result = np.full(n, np.nan)
    min_periods = max(2, T // 2)

    for i in range(n):
        rd_i = report_dates[i]

        # Collect PIT-safe indices: index <= i and report_date <= rd_i
        pit_indices = [j for j in range(i + 1) if report_dates[j] <= rd_i]
        if not pit_indices:
            continue

        # Use last T PIT-safe records
        window_idx = pit_indices[-T:]
        window_g = g_vals[window_idx]

        # Filter out nan and inf
        valid_mask = ~np.isnan(window_g) & ~np.isinf(window_g)
        valid_g = window_g[valid_mask]

        if len(valid_g) < min_periods:
            continue

        mean_g = valid_g.mean()
        if abs(mean_g) < MIN_ABS_MEAN_G:
            continue

        # Time index for valid records (1-based positions within window)
        t_full = np.arange(1, len(window_g) + 1, dtype=np.float64)
        t_valid = t_full[valid_mask]

        if len(t_valid) < 2:
            continue

        # OLS slope: b = cov(t, g) / var(t)
        t_mean = t_valid.mean()
        cov_tg = np.sum((t_valid - t_mean) * (valid_g - mean_g))
        var_t = np.sum((t_valid - t_mean) ** 2)
        if abs(var_t) < 1e-10:
            continue

        b = cov_tg / var_t
        result[i] = b / mean_g

    return result


def _build_quarterly_panel(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Extract quarterly records for a field, deduplicate, sort by fiscal date,
    and return as a DataFrame with columns: symbol, date, report_date, <field>.
    """
    needed = ["symbol", "date", "report_date", field]
    df = raw[needed].dropna(subset=[field]).copy()

    # Sanity: fiscal date must not exceed disclosure date
    df = df[df["date"] <= df["report_date"]]

    # Deduplicate: same (symbol, fiscal date) -> keep earliest report_date
    df = df.sort_values(["symbol", "date", "report_date"])
    df = df.groupby(["symbol", "date"], as_index=False).first()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


def _build_daily_panel_from_quarterly(
    qdf: pd.DataFrame,
    col: str,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a daily forward-filled panel (symbol x trading_dates) from
    quarterly computed values, using report_date as the availability date.
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


def _cross_section_zscore(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cross-sectional z-score normalization at each date column.
    Returns a DataFrame of the same shape.
    """
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
    """
    Cross-sectional rank normalization to [0, 1] at each date.
    NaN positions remain NaN.
    """
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


class COMP_ACCEL(FundamentalFactorCalculator):
    """
    Composite Growth Acceleration (COMP_ACCEL) factor.

    For each of 4 fundamental fields and 2 time windows, fits a linear trend
    on the YoY growth rate sequence and computes acc_comp = b / mean(g).
    Synthesizes via z-score, equal-weight average, and rank normalization.
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
            "fields": COMP_ACCEL_FIELDS,
            "windows": COMP_ACCEL_WINDOWS,
            "g_winsor": [G_WINSOR_LO, G_WINSOR_HI],
            "min_abs_mean_g": MIN_ABS_MEAN_G,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute COMP_ACCEL daily panel (N stocks x T_days).

        Steps:
          1. Load raw quarterly data for all 4 fields.
          2. Per field: winsorize g, compute COMP_ACCEL for each symbol at T=4 and T=8.
          3. Forward-fill each (field, T) COMP_ACCEL panel to daily via report_date.
          4. Cross-sectional z-score each daily panel.
          5. Equal-weight average across all (field, T) pairs.
          6. Cross-sectional rank normalize to [0, 1].
          7. Filter to mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        end_date = fundamental_data.end_date

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("COMP_ACCEL: no raw fundamental data available")

        raw = fundamental_data._raw_data

        # Collect z-scored daily panels for each (field, T) combination
        zscored_panels = []

        for field in COMP_ACCEL_FIELDS:
            # Step 1: Extract and deduplicate quarterly data for this field
            qdf = _build_quarterly_panel(raw, field)
            if qdf.empty:
                continue

            # Step 2: Winsorize g values
            qdf = qdf.copy()
            qdf[field] = qdf[field].clip(G_WINSOR_LO, G_WINSOR_HI)

            for T in COMP_ACCEL_WINDOWS:
                accel_col = f"comp_accel_{field}_T{T}"
                accel_vals = np.full(len(qdf), np.nan)

                # Compute COMP_ACCEL per symbol
                for symbol, grp in qdf.groupby("symbol", sort=False):
                    grp_sorted = grp.sort_values("date")
                    idx = grp_sorted.index
                    g_arr = grp_sorted[field].values.astype(float)
                    rd_arr = grp_sorted["report_date"].values.astype("datetime64[ns]")
                    acc_arr = _compute_comp_accel_quarterly(g_arr, T, rd_arr)
                    accel_vals[idx] = acc_arr

                qdf[accel_col] = accel_vals

                # Step 3: Forward-fill to daily panel
                daily_panel = _build_daily_panel_from_quarterly(
                    qdf, accel_col, trading_dates, end_date
                )
                if daily_panel.empty or daily_panel.shape[0] == 0:
                    continue

                # Step 4: Cross-sectional z-score
                zs_panel = _cross_section_zscore(daily_panel)
                zscored_panels.append(zs_panel)

        if not zscored_panels:
            raise ValueError("COMP_ACCEL: no valid sub-panels computed")

        # Step 5: Align all panels to union index/columns and equal-weight average
        all_symbols = sorted(set().union(*[set(p.index.tolist()) for p in zscored_panels]))
        all_dates = trading_dates  # all panels already share the same trading_dates

        aligned = []
        for p in zscored_panels:
            aligned.append(p.reindex(index=all_symbols, columns=all_dates))

        stacked = np.stack([a.values for a in aligned], axis=0)  # (K, N, T)
        valid_count = np.sum(~np.isnan(stacked), axis=0)          # (N, T)
        composite = np.nanmean(stacked, axis=0)                    # (N, T)
        composite[valid_count == 0] = np.nan

        # Step 6: Cross-sectional rank normalization
        composite = _cross_section_rank_norm(composite)

        # Step 7: Mainboard filter
        symbols_arr = np.array(all_symbols)
        mb_mask = np.array([_is_mainboard(s) for s in symbols_arr])
        composite = composite[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        if len(symbols_arr) == 0:
            raise ValueError("COMP_ACCEL: no mainboard stocks after filtering")

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(f"COMP_ACCEL NaN ratio is high: {nan_ratio:.1%}, please check data")

        dates_arr = np.array(all_dates, dtype="datetime64[ns]")

        return FactorData(
            values=composite.astype(np.float64),
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python comp_accel.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("COMP_ACCEL factor smoke test")
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

    print(f"\n[Step 2] Compute COMP_ACCEL factor")
    calculator = COMP_ACCEL()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.shape}")
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
    else:
        print("  No valid values in last cross-section")

    # --- Section 4.1 Smoke Test Assertions ---
    print(f"\n[Step 3] Smoke test assertions")

    assert result.values.ndim == 2, "values must be 2-D"
    print(f"  [PASS] shape is 2-D: {result.values.shape}")

    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"
    print(f"  [PASS] dtype = float64")

    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"  [PASS] NaN ratio < 80%: {nan_ratio:.1%}")

    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"
    print(f"  [PASS] No inf values")

    # Idempotency
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
