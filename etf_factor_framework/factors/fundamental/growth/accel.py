"""
ACCEL factor (Growth Acceleration - Second-Order Growth Rate)

Computes: acc = (g_t - g_{t-T}) / |g_{t-T}|
where g is the YoY growth rate (already available as a field in lixinger).

Measures the rate of change in growth itself (acceleration).
acc > 0: growth momentum strengthening; acc < 0: growth momentum weakening.

Two time windows (T in quarters):
  - T=4  (1-year window: current vs 4 quarters ago)
  - T=20 (5-year window: current vs 20 quarters ago)

Applied to 4 YoY growth rate fields:
  - q_ps_toi_c_y2y       : total operating income YoY growth
  - q_ps_npatoshopc_c_y2y: attributable net profit YoY growth
  - q_ps_op_c_y2y        : operating profit YoY growth
  - q_cfs_ncffoa_c_y2y   : operating cash flow net YoY growth

Synthesis:
  1. Winsorize g to [-5, 5] before computing ACCEL.
  2. If |g_{t-T}| < 0.01, set ACCEL to NaN (denominator too small).
  3. For each (field, T) pair, z-score cross-sectionally at each date.
  4. Equal-weight average across all (field, T) combinations (max 8).
  5. Cross-sectional rank normalization to [0, 1].

Data source : lixinger.financial_statements (fs_type='q')
Factor direction : positive (higher acceleration = stronger momentum)
Factor category  : growth - acceleration method

Notes:
  - All data is PIT-safe: ACCEL for a given quarter is available at report_date.
  - Output is restricted to A-share mainboard stocks (60xxxx / 00xxxx).
  - T-quarter lag is computed positionally (shift by T quarters in sorted order),
    not by calendar date, to handle irregular reporting intervals.
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

FACTOR_NAME = "ACCEL"
FACTOR_DIRECTION = 1  # positive: higher acceleration is better

# Fields to compute ACCEL on
ACCEL_FIELDS = [
    "q_ps_toi_c_y2y",
    "q_ps_npatoshopc_c_y2y",
    "q_ps_op_c_y2y",
    "q_cfs_ncffoa_c_y2y",
]

# Time windows (in quarters)
ACCEL_WINDOWS = [4, 20]

# Winsorization bounds for g values
G_WINSOR_LO = -5.0
G_WINSOR_HI = 5.0

# Minimum absolute denominator to avoid explosion
MIN_ABS_DENOM = 0.01


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _compute_accel_quarterly(
    g_vals: np.ndarray,
    T: int,
    fiscal_dates: np.ndarray,
    report_dates: np.ndarray,
) -> np.ndarray:
    """
    Compute ACCEL for one symbol's quarterly g series using calendar-based
    quarter matching (NOT positional shifting).

    Calendar-based matching: for each record at fiscal_date D, look up the
    record whose fiscal_date is exactly T quarters before D (by date value),
    rather than the record T positions back in the array. This prevents
    positional index errors when intermediate quarterly records are missing
    (e.g., late-filed reports excluded in truncated datasets).

    PIT-safety: ACCEL at record i is set to NaN if the lookback record's
    report_date is later than record i's report_date.

    Args:
        g_vals       : float64 array (n,) of winsorized YoY growth rates
        T            : int, lookback window in quarters
        fiscal_dates : datetime64[ns] array (n,) of fiscal period end dates
        report_dates : datetime64[ns] array (n,) of publication dates

    Returns:
        float64 array (n,) of ACCEL values
    """
    n = len(g_vals)
    result = np.full(n, np.nan)

    # Build a lookup: fiscal_date -> (g_val, report_date)
    fd_to_idx = {}
    for k in range(n):
        fd_to_idx[fiscal_dates[k]] = k

    for i in range(n):
        g_t = g_vals[i]
        if np.isnan(g_t):
            continue

        # Target fiscal date: T quarters before fiscal_dates[i]
        fd_i = pd.Timestamp(fiscal_dates[i])
        fd_back = fd_i - pd.DateOffset(months=T * 3)
        # Normalize to quarter-end (handles edge cases)
        fd_back_end = fd_back + pd.offsets.QuarterEnd(0)
        target_key = np.datetime64(fd_back_end.normalize(), 'ns')

        if target_key not in fd_to_idx:
            continue
        j = fd_to_idx[target_key]

        g_tT = g_vals[j]
        if np.isnan(g_tT):
            continue

        # PIT check: historical record must have been published by the current report_date
        if report_dates[j] > report_dates[i]:
            continue

        abs_denom = abs(g_tT)
        if abs_denom < MIN_ABS_DENOM:
            continue

        result[i] = (g_t - g_tT) / abs_denom

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


class ACCEL(FundamentalFactorCalculator):
    """
    Growth Acceleration (ACCEL) factor.

    Measures the acceleration of YoY growth rates across 4 fundamental metrics
    over 2 time windows (1-year and 5-year), synthesized via z-score and rank.
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
            "fields": ACCEL_FIELDS,
            "windows": ACCEL_WINDOWS,
            "g_winsor": [G_WINSOR_LO, G_WINSOR_HI],
            "min_abs_denom": MIN_ABS_DENOM,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute ACCEL daily panel (N stocks x T_days).

        Steps:
          1. Load raw quarterly data for all 4 fields.
          2. Per field: winsorize g, compute ACCEL for each symbol at T=4 and T=20.
          3. Forward-fill each (field, T) ACCEL panel to daily via report_date.
          4. Cross-sectional z-score each daily panel.
          5. Equal-weight average across all (field, T) pairs.
          6. Cross-sectional rank normalize to [0, 1].
          7. Filter to mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        end_date = fundamental_data.end_date

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("ACCEL: no raw fundamental data available")

        raw = fundamental_data._raw_data

        # Collect z-scored daily panels for each (field, T) combination
        zscored_panels = []

        for field in ACCEL_FIELDS:
            # Step 1: Extract and deduplicate quarterly data for this field
            qdf = _build_quarterly_panel(raw, field)
            if qdf.empty:
                continue

            # Step 2: Winsorize g values
            qdf = qdf.copy()
            qdf[field] = qdf[field].clip(G_WINSOR_LO, G_WINSOR_HI)

            for T in ACCEL_WINDOWS:
                accel_col = f"accel_{field}_T{T}"
                accel_vals = np.full(len(qdf), np.nan)

                # Compute ACCEL per symbol
                for symbol, grp in qdf.groupby("symbol", sort=False):
                    grp_sorted = grp.sort_values("date")
                    idx = grp_sorted.index
                    g_arr = grp_sorted[field].values.astype(float)
                    fd_arr = grp_sorted["date"].values.astype("datetime64[ns]")
                    rd_arr = grp_sorted["report_date"].values.astype("datetime64[ns]")
                    acc_arr = _compute_accel_quarterly(g_arr, T, fd_arr, rd_arr)
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
            raise ValueError("ACCEL: no valid sub-panels computed")

        # Step 5: Align all panels to union index/columns and equal-weight average
        all_symbols = sorted(set().union(*[set(p.index.tolist()) for p in zscored_panels]))
        all_dates = trading_dates  # all panels already share the same trading_dates

        aligned = []
        for p in zscored_panels:
            aligned.append(p.reindex(index=all_symbols, columns=all_dates))

        stacked = np.stack([a.values for a in aligned], axis=0)  # (K, N, T)
        valid_count = np.sum(~np.isnan(stacked), axis=0)          # (N, T)
        composite = np.nanmean(stacked, axis=0)                    # (N, T)
        # If no valid sub-panel contributes, set to NaN
        composite[valid_count == 0] = np.nan

        # Step 6: Cross-sectional rank normalization
        composite = _cross_section_rank_norm(composite)

        # Step 7: Mainboard filter
        symbols_arr = np.array(all_symbols)
        mb_mask = np.array([_is_mainboard(s) for s in symbols_arr])
        composite = composite[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        if len(symbols_arr) == 0:
            raise ValueError("ACCEL: no mainboard stocks after filtering")

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(f"ACCEL NaN ratio is high: {nan_ratio:.1%}, please check data")

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
# Smoke test (run: python accel.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("ACCEL factor smoke test")
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

    print(f"\n[Step 2] Compute ACCEL factor")
    calculator = ACCEL()
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
