"""
OCF_GROWTH_COMP factor (Operating Cash Flow Growth Composite)

Computes a composite of 8 sub-factors, all based on operating cash flow
(q_cfs_ncffoa_c) and its YoY growth rate (q_cfs_ncffoa_c_y2y):

  1. PCT_GROWTH T=2   (6-month pct growth)
  2. PCT_GROWTH T=4   (1-year YoY pct growth)
  3. PCT_GROWTH T=20  (5-year pct growth)
  4. SURPRISE_GROWTH T=8  (2-year SUE-style)
  5. ACCEL T=4        (1-year acceleration of YoY rate)
  6. ACCEL T=20       (5-year acceleration of YoY rate)
  7. ROBUST_GROWTH T=4  (1-year growth / std)
  8. ROBUST_GROWTH T=20 (5-year growth / std)

Synthesis:
  - Each sub-factor is cross-sectionally Winsorized (1-99%) and z-scored daily.
  - Equal-weight nanmean; at least 4 valid sub-factors required per (stock, date).
  - Cross-sectional rank normalization to [0, 1].
  - Restricted to A-share mainboard stocks (60xxxx / 00xxxx).

Data source : lixinger.financial_statements
Factor direction : positive (higher composite OCF growth is better)
Factor category  : growth - OCF composite multi-method

Notes:
  - PCT_GROWTH denominator = |x_{t-T}|; near-zero (<1 yuan) -> NaN.
  - SURPRISE_GROWTH requires >= 4 valid historical quarters; std=0 -> NaN.
  - ACCEL uses q_cfs_ncffoa_c_y2y winsorized to [-5, 5];
    |g_{t-T}| < 0.01 -> NaN; calendar-based quarter matching.
  - ROBUST_GROWTH requires T+1 contiguous PIT-valid quarters; std=0 -> NaN.
  - All computations are PIT-safe: only data with report_date <= current
    report_date enters any rolling window.
  - Output restricted to A-share mainboard (60xxxx / 00xxxx).
  - OCF can be negative; PCT_GROWTH denominator uses absolute value to
    prevent sign distortion.
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

FACTOR_NAME = "OCF_GROWTH_COMP"
FACTOR_DIRECTION = 1  # positive

# Fields
OCF_FIELD = "q_cfs_ncffoa_c"          # single-quarter operating cash flow
OCF_Y2Y_FIELD = "q_cfs_ncffoa_c_y2y"  # pre-computed YoY growth rate

# Thresholds
MIN_ABS_DENOM_PCT = 1.0     # yuan, for PCT_GROWTH denominator guard
MIN_HIST_PERIODS_SURPRISE = 4  # minimum valid lags for SURPRISE_GROWTH
ACCEL_WINSOR_LO = -5.0      # g winsorization lower bound (ACCEL)
ACCEL_WINSOR_HI = 5.0       # g winsorization upper bound (ACCEL)
MIN_ABS_DENOM_ACCEL = 0.01  # minimum |g_{t-T}| for ACCEL
WINSOR_LO_PCT = 1.0         # cross-section winsorization percentile (low)
WINSOR_HI_PCT = 99.0        # cross-section winsorization percentile (high)
MIN_VALID_SUBS = 4           # minimum valid sub-factors for composite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _build_quarterly_panel(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Extract quarterly records for field; deduplicate and sort.

    Deduplication: same (symbol, fiscal date) -> keep earliest report_date.
    Sanity filter: fiscal date must not exceed its disclosure date.
    """
    needed = ["symbol", "date", "report_date", field]
    df = raw[needed].dropna(subset=[field]).copy()
    df = df[df["date"] <= df["report_date"]]
    df = df.sort_values(["symbol", "date", "report_date"])
    df = df.groupby(["symbol", "date"], as_index=False).first()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _build_daily_panel(
    df: pd.DataFrame,
    col: str,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Forward-fill quarterly col to a daily panel (symbol x trading_dates)
    using report_date as the signal availability date.
    """
    sub = df[["symbol", "report_date", col]].dropna(subset=[col]).copy()
    sub = sub[sub["report_date"] <= end_date]
    if sub.empty:
        return None
    pivot = sub.pivot_table(
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
        if valid_mask.sum() < 10:
            continue
        lo = np.percentile(col[valid_mask], WINSOR_LO_PCT)
        hi = np.percentile(col[valid_mask], WINSOR_HI_PCT)
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


# ---------------------------------------------------------------------------
# Sub-factor computation functions
# ---------------------------------------------------------------------------

def _compute_pct_growth_panel(
    qdf: pd.DataFrame,
    T: int,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Compute PCT_GROWTH = (x_t - x_{t-T}) / |x_{t-T}| for OCF_FIELD over T quarters.

    Uses fiscal date shift by T*3 months; PIT filter on lag report_date.
    Denominator uses absolute value since OCF can be negative.
    """
    if qdf.empty:
        return None

    qdf_idx = qdf.set_index(["symbol", "date"])
    val_lookup = qdf_idx[OCF_FIELD]
    rd_lookup = qdf_idx["report_date"]

    lag_dates = qdf["date"] - pd.DateOffset(months=3 * T)
    midx = pd.MultiIndex.from_arrays([qdf["symbol"], lag_dates])
    lag_val = val_lookup.reindex(midx).values.astype(float)
    lag_rd = rd_lookup.reindex(midx).values

    cur_rd = qdf["report_date"].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pit_ok = np.array([
            (lrd is not pd.NaT
             and not pd.isnull(lrd)
             and lrd <= crd)
            for lrd, crd in zip(lag_rd, cur_rd)
        ])

    lag_val_pit = np.where(pit_ok, lag_val, np.nan)

    x_t = qdf[OCF_FIELD].values.astype(float)
    abs_denom = np.abs(lag_val_pit)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        growth = (x_t - lag_val_pit) / abs_denom

    # Guard: near-zero denominator or NaN inputs
    growth = np.where(abs_denom >= MIN_ABS_DENOM_PCT, growth, np.nan)
    growth = np.where(~np.isnan(x_t) & ~np.isnan(lag_val_pit), growth, np.nan)

    col_name = f"pct_T{T}"
    qdf = qdf.copy()
    qdf[col_name] = growth

    return _build_daily_panel(qdf, col_name, trading_dates, end_date)


def _compute_surprise_growth_panel(
    qdf: pd.DataFrame,
    T: int,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Compute SURPRISE_GROWTH = (x_t - mean(x_{t-1}..x_{t-T})) / std(x_{t-1}..x_{t-T}).

    Requires >= MIN_HIST_PERIODS valid lags; std=0 -> NaN.
    """
    if qdf.empty:
        return None

    qdf_idx = qdf.set_index(["symbol", "date"])
    val_lookup = qdf_idx[OCF_FIELD]
    rd_lookup = qdf_idx["report_date"]

    n_rows = len(qdf)
    hist_vals = np.full((n_rows, T), np.nan)

    for k in range(1, T + 1):
        lag_dates = qdf["date"] - pd.DateOffset(months=3 * k)
        midx = pd.MultiIndex.from_arrays([qdf["symbol"], lag_dates])
        lag_val = val_lookup.reindex(midx).values.astype(float)
        lag_rd = rd_lookup.reindex(midx).values
        cur_rd = qdf["report_date"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pit_ok = np.array([
                (lrd is not pd.NaT
                 and not pd.isnull(lrd)
                 and lrd <= crd)
                for lrd, crd in zip(lag_rd, cur_rd)
            ])

        hist_vals[:, k - 1] = np.where(pit_ok, lag_val, np.nan)

    n_valid = np.sum(~np.isnan(hist_vals), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hist_mean = np.nanmean(hist_vals, axis=1)
        hist_std = np.nanstd(hist_vals, axis=1, ddof=1)

    x_t = qdf[OCF_FIELD].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_surprise = (x_t - hist_mean) / hist_std

    invalid = (
        (n_valid < MIN_HIST_PERIODS_SURPRISE)
        | (hist_std == 0)
        | np.isnan(x_t)
        | np.isnan(hist_mean)
    )
    g_surprise = np.where(invalid, np.nan, g_surprise)

    col_name = f"surprise_T{T}"
    qdf = qdf.copy()
    qdf[col_name] = g_surprise

    return _build_daily_panel(qdf, col_name, trading_dates, end_date)


def _compute_accel_quarterly(
    g_vals: np.ndarray,
    T: int,
    fiscal_dates: np.ndarray,
    report_dates: np.ndarray,
) -> np.ndarray:
    """
    Compute ACCEL for one symbol using calendar-based quarter matching.

    PIT-safe: historical record must be published by current report_date.
    """
    n = len(g_vals)
    result = np.full(n, np.nan)

    fd_to_idx = {}
    for k in range(n):
        fd_to_idx[fiscal_dates[k]] = k

    for i in range(n):
        g_t = g_vals[i]
        if np.isnan(g_t):
            continue

        fd_i = pd.Timestamp(fiscal_dates[i])
        fd_back = fd_i - pd.DateOffset(months=T * 3)
        fd_back_end = fd_back + pd.offsets.QuarterEnd(0)
        target_key = np.datetime64(fd_back_end.normalize(), 'ns')

        if target_key not in fd_to_idx:
            continue
        j = fd_to_idx[target_key]

        g_tT = g_vals[j]
        if np.isnan(g_tT):
            continue

        if report_dates[j] > report_dates[i]:
            continue

        abs_denom = abs(g_tT)
        if abs_denom < MIN_ABS_DENOM_ACCEL:
            continue

        result[i] = (g_t - g_tT) / abs_denom

    return result


def _compute_accel_panel(
    qdf_y2y: pd.DataFrame,
    T: int,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Compute ACCEL sub-factor for q_cfs_ncffoa_c_y2y with window T.
    """
    if qdf_y2y.empty:
        return None

    qdf_y2y = qdf_y2y.copy()
    # Winsorize g values to [-5, 5]
    qdf_y2y[OCF_Y2Y_FIELD] = qdf_y2y[OCF_Y2Y_FIELD].clip(
        ACCEL_WINSOR_LO, ACCEL_WINSOR_HI
    )

    col_name = f"accel_T{T}"
    accel_vals = np.full(len(qdf_y2y), np.nan)

    for symbol, grp in qdf_y2y.groupby("symbol", sort=False):
        grp_sorted = grp.sort_values("date")
        idx = grp_sorted.index
        g_arr = grp_sorted[OCF_Y2Y_FIELD].values.astype(float)
        fd_arr = grp_sorted["date"].values.astype("datetime64[ns]")
        rd_arr = grp_sorted["report_date"].values.astype("datetime64[ns]")
        acc_arr = _compute_accel_quarterly(g_arr, T, fd_arr, rd_arr)
        accel_vals[idx] = acc_arr

    qdf_y2y[col_name] = accel_vals

    return _build_daily_panel(qdf_y2y, col_name, trading_dates, end_date)


def _compute_robust_growth_series(
    vals_arr: np.ndarray,
    rd_arr: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Compute ROBUST_GROWTH = (x_t - x_{t-T}) / std(x_{t-T..t}) for one symbol.

    PIT-safe: uses only quarters with report_date <= current report_date in window.
    Requires T+1 PIT-valid data points; std=0 -> NaN.
    """
    n = len(vals_arr)
    results = np.full(n, np.nan)

    for i in range(n):
        cur_rd = rd_arr[i]
        pit_ok = rd_arr[:i + 1] <= cur_rd
        pit_indices = np.where(pit_ok)[0]

        if len(pit_indices) < T + 1:
            continue

        window_indices = pit_indices[-(T + 1):]
        x = vals_arr[window_indices].astype(float)

        if np.any(np.isnan(x)):
            continue

        x_t = x[-1]
        x_tT = x[0]
        std_val = np.std(x, ddof=0)

        if std_val == 0.0:
            continue

        results[i] = (x_t - x_tT) / std_val

    return results


def _compute_robust_growth_panel(
    qdf: pd.DataFrame,
    T: int,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """Compute ROBUST_GROWTH sub-factor for OCF_FIELD with window T."""
    if qdf.empty:
        return None

    col_name = f"robust_T{T}"
    rg_vals = np.full(len(qdf), np.nan)

    for symbol, grp in qdf.groupby("symbol", sort=False):
        grp_sorted = grp.sort_values("date")
        idx = grp_sorted.index
        vals = grp_sorted[OCF_FIELD].values.astype(float)
        rd_arr = grp_sorted["report_date"].values.astype("datetime64[ns]")
        rg_arr = _compute_robust_growth_series(vals, rd_arr, T)
        rg_vals[idx] = rg_arr

    qdf = qdf.copy()
    qdf[col_name] = rg_vals

    return _build_daily_panel(qdf, col_name, trading_dates, end_date)


# ---------------------------------------------------------------------------
# Factor class
# ---------------------------------------------------------------------------

class OCF_GROWTH_COMP(FundamentalFactorCalculator):
    """
    Operating Cash Flow Growth Composite factor.

    Combines 8 sub-factors for net operating cash flow growth:
      PCT_GROWTH (T=2, 4, 20), SURPRISE_GROWTH (T=8),
      ACCEL (T=4, 20), ROBUST_GROWTH (T=4, 20).

    Each sub-factor is independently winsorized and z-scored cross-sectionally,
    then equal-weighted into the composite (>= 4 valid sub-factors required).
    Final output is rank-normalized to [0, 1] and restricted to mainboard stocks.
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
            "ocf_field": OCF_FIELD,
            "ocf_y2y_field": OCF_Y2Y_FIELD,
            "sub_factors": [
                "pct_T2", "pct_T4", "pct_T20",
                "surprise_T8",
                "accel_T4", "accel_T20",
                "robust_T4", "robust_T20",
            ],
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute OCF_GROWTH_COMP daily panel (N_stocks x T_days).

        Steps:
          1. Load raw fundamental data.
          2. Build deduplicated quarterly panels for OCF_FIELD and OCF_Y2Y_FIELD.
          3. Compute 8 sub-factor daily panels (PCT x3, SURPRISE x1, ACCEL x2,
             ROBUST x2).
          4. Per sub-factor: cross-section winsorize + z-score.
          5. Equal-weight average (need >= MIN_VALID_SUBS valid per cell).
          6. Cross-section rank normalize to [0, 1].
          7. Filter to A-share mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        end_date = fundamental_data.end_date

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("OCF_GROWTH_COMP: no raw fundamental data available")

        raw = fundamental_data._raw_data

        # Step 2: Build deduplicated quarterly panels
        qdf_ocf = _build_quarterly_panel(raw, OCF_FIELD)
        qdf_y2y = _build_quarterly_panel(raw, OCF_Y2Y_FIELD) if OCF_Y2Y_FIELD in raw.columns else pd.DataFrame()

        # Step 3: Compute sub-factor daily panels
        sub_panels = []
        sub_names = []

        # PCT_GROWTH: T=2, 4, 20
        for T in [2, 4, 20]:
            panel = _compute_pct_growth_panel(qdf_ocf, T, trading_dates, end_date)
            if panel is not None and not panel.empty:
                sub_panels.append(panel)
                sub_names.append(f"pct_T{T}")

        # SURPRISE_GROWTH: T=8
        panel = _compute_surprise_growth_panel(qdf_ocf, 8, trading_dates, end_date)
        if panel is not None and not panel.empty:
            sub_panels.append(panel)
            sub_names.append("surprise_T8")

        # ACCEL: T=4, 20 (uses OCF_Y2Y_FIELD)
        if not qdf_y2y.empty:
            for T in [4, 20]:
                panel = _compute_accel_panel(qdf_y2y, T, trading_dates, end_date)
                if panel is not None and not panel.empty:
                    sub_panels.append(panel)
                    sub_names.append(f"accel_T{T}")
        else:
            warnings.warn(
                f"OCF_GROWTH_COMP: field '{OCF_Y2Y_FIELD}' not available; "
                "ACCEL sub-factors skipped."
            )

        # ROBUST_GROWTH: T=4, 20
        for T in [4, 20]:
            panel = _compute_robust_growth_panel(qdf_ocf, T, trading_dates, end_date)
            if panel is not None and not panel.empty:
                sub_panels.append(panel)
                sub_names.append(f"robust_T{T}")

        if len(sub_panels) < MIN_VALID_SUBS:
            raise ValueError(
                f"OCF_GROWTH_COMP: only {len(sub_panels)} valid sub-panels "
                f"computed, need >= {MIN_VALID_SUBS}"
            )

        # Step 4: Winsorize + z-score each sub-panel
        normed_panels = []
        for panel in sub_panels:
            p = _cross_section_winsorize_panel(panel)
            p = _cross_section_zscore(p)
            normed_panels.append(p)

        # Step 5: Align to union symbols and equal-weight average
        all_symbols = sorted(
            set().union(*[set(p.index.tolist()) for p in normed_panels])
        )

        aligned = [
            p.reindex(index=all_symbols, columns=trading_dates)
            for p in normed_panels
        ]
        stacked = np.stack([a.values for a in aligned], axis=0)  # (K, N, T_days)
        valid_count = np.sum(~np.isnan(stacked), axis=0)          # (N, T_days)
        composite = np.nanmean(stacked, axis=0)                    # (N, T_days)
        composite[valid_count < MIN_VALID_SUBS] = np.nan

        # Step 6: Cross-section rank normalization
        composite = _cross_section_rank_norm(composite)

        # Step 7: Mainboard filter
        symbols_arr = np.array(all_symbols)
        mb_mask = np.array([_is_mainboard(s) for s in symbols_arr])
        composite = composite[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        if len(symbols_arr) == 0:
            raise ValueError("OCF_GROWTH_COMP: no mainboard stocks after filtering")

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(f"OCF_GROWTH_COMP NaN ratio is high: {nan_ratio:.1%}")

        dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        return FactorData(
            values=composite.astype(np.float64),
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ---------------------------------------------------------------------------
# Smoke test  (run: python ocf_growth_comp.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("OCF_GROWTH_COMP factor smoke test")
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

    print(f"\n[Step 2] Compute OCF_GROWTH_COMP factor")
    calculator = OCF_GROWTH_COMP()
    result = calculator.calculate(fd)

    print(f"\nFactor shape   : {result.shape}")
    print(f"Symbols        : {result.symbols}")
    print(f"Date range     : "
          f"{pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

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

    assert result.values.dtype == np.float64, \
        f"expected float64, got {result.values.dtype}"
    print(f"  [PASS] dtype = float64")

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"  [PASS] NaN ratio < 80%: {nan_ratio:.1%}")

    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"
    print(f"  [PASS] No inf values")

    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), \
        "Idempotency failed"
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
