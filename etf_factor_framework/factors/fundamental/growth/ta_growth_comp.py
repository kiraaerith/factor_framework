"""
TA_GROWTH_COMP factor (Total Assets Growth Composite)

Computes a composite of 2 sub-factors based on total assets (q_bs_ta_t):

  1. SURPRISE_GROWTH T=8   (2-year SUE-style: deviation from 8-quarter mean)
  2. ROBUST_GROWTH   T=20  (5-year stable growth: 20-quarter delta / std)

Synthesis:
  - Each sub-factor is cross-sectionally Winsorized (1-99%) and z-scored daily.
  - Equal-weight nanmean; at least 1 valid sub-factor required per (stock, date).
  - Cross-sectional rank normalization to [0, 1].
  - Restricted to A-share mainboard stocks (60xxxx / 00xxxx).

Data source : lixinger.financial_statements  (q_bs_ta_t)
Factor direction : positive (higher composite TA growth is better)
Factor category  : growth - capital investment dimension

Design rationale:
  Total assets is a balance-sheet time-point value; its series is smoother
  than income-statement flow items. PCT_GROWTH and ACCEL add little
  incremental information on smooth series, so TA_GROWTH_COMP uses only
  SURPRISE_GROWTH (deviation-from-mean for recency signal) and ROBUST_GROWTH
  (long-run delta-per-unit-volatility for sustained expansion).

Notes:
  - q_bs_ta_t <= 0: data quality issue; set to NaN before computation.
  - SURPRISE_GROWTH: requires >= 4 valid historical quarters; std=0 -> NaN.
  - ROBUST_GROWTH: requires >= 11 of 21 window points; endpoints non-NaN;
    std=0 -> NaN; uses ddof=1.
  - All computations are PIT-safe: historical records must be disclosed
    (report_date) on or before the current report_date.
  - Output restricted to A-share mainboard (60xxxx / 00xxxx).
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

FACTOR_NAME = "TA_GROWTH_COMP"
FACTOR_DIRECTION = 1  # positive

# Field
TA_FIELD = "q_bs_ta_t"  # total assets, balance sheet time-point value

# Sub-factor windows
T_SURPRISE = 8   # 8 quarters ~ 2 years
T_ROBUST   = 20  # 20 quarters ~ 5 years

# Thresholds
MIN_HIST_PERIODS_SURPRISE = 4   # minimum valid lag quarters for SURPRISE_GROWTH
MIN_VALID_ROBUST = 11           # minimum valid points in 21-point ROBUST window
WINSOR_LO_PCT = 1.0             # cross-section winsorization percentile (low)
WINSOR_HI_PCT = 99.0            # cross-section winsorization percentile (high)
MIN_VALID_SUBS = 1              # composite valid if at least 1 sub-factor is valid


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
    Positivity filter for TA_FIELD: values <= 0 set to NaN (data quality).
    """
    needed = ["symbol", "date", "report_date", field]
    df = raw[needed].dropna(subset=[field]).copy()
    df = df[df["date"] <= df["report_date"]]
    df = df.sort_values(["symbol", "date", "report_date"])
    df = df.groupby(["symbol", "date"], as_index=False).first()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    if field == TA_FIELD:
        # Total assets should always be positive; non-positive values are data errors
        non_positive_mask = df[field] <= 0
        if non_positive_mask.any():
            warnings.warn(
                f"TA_GROWTH_COMP: {non_positive_mask.sum()} non-positive {field} "
                "values found; setting to NaN"
            )
            df.loc[non_positive_mask, field] = np.nan
        df = df.dropna(subset=[field])

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

def _compute_surprise_growth_panel(
    qdf: pd.DataFrame,
    T: int,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Compute SURPRISE_GROWTH = (x_t - mean(x_{t-1}..x_{t-T})) / std(x_{t-1}..x_{t-T}).

    Uses 8 quarterly lags (k=1..T). PIT check: lag report_date <= current report_date.
    Requires >= MIN_HIST_PERIODS_SURPRISE valid lags; std=0 -> NaN (ddof=1).
    """
    if qdf.empty:
        return None

    qdf_idx = qdf.set_index(["symbol", "date"])
    val_lookup = qdf_idx[TA_FIELD]
    rd_lookup = qdf_idx["report_date"]

    n_rows = len(qdf)
    hist_vals = np.full((n_rows, T), np.nan)
    cur_rd = qdf["report_date"].values

    for k in range(1, T + 1):
        lag_dates = qdf["date"] - pd.DateOffset(months=3 * k)
        midx = pd.MultiIndex.from_arrays([qdf["symbol"], lag_dates])
        lag_val = val_lookup.reindex(midx).values.astype(float)
        lag_rd = rd_lookup.reindex(midx).values

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

    x_t = qdf[TA_FIELD].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_surprise = (x_t - hist_mean) / hist_std

    invalid = (
        (n_valid < MIN_HIST_PERIODS_SURPRISE)
        | (hist_std == 0)
        | np.isnan(x_t)
        | np.isnan(hist_mean)
        | np.isnan(hist_std)
    )
    g_surprise = np.where(invalid, np.nan, g_surprise)

    col_name = f"surprise_T{T}"
    qdf = qdf.copy()
    qdf[col_name] = g_surprise

    return _build_daily_panel(qdf, col_name, trading_dates, end_date)


def _compute_robust_growth_panel(
    qdf: pd.DataFrame,
    T: int,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Compute ROBUST_GROWTH = (x_t - x_{t-T}) / std(x_{t-T}..x_t).

    Window of T+1 points (k=0..T). k=0 is current; k=T is x_{t-T}.
    PIT check for k>=1: lag report_date <= current report_date.
    Requires:
      - >= MIN_VALID_ROBUST valid points in window
      - both endpoints (x_{t-T} and x_t) non-NaN
      - std > 0 (ddof=1, using nanstd to tolerate interior NaNs)
    """
    if qdf.empty:
        return None

    qdf_idx = qdf.set_index(["symbol", "date"])
    val_lookup = qdf_idx[TA_FIELD]
    rd_lookup = qdf_idx["report_date"]

    n_rows = len(qdf)
    # window_vals[:, 0] = x_{t-T} (oldest), window_vals[:, T] = x_t (current)
    window_vals = np.full((n_rows, T + 1), np.nan)
    cur_rd = qdf["report_date"].values

    # k=0: current period (x_t)
    window_vals[:, T] = qdf[TA_FIELD].values.astype(float)

    # k=1..T: historical lags
    for k in range(1, T + 1):
        lag_dates = qdf["date"] - pd.DateOffset(months=3 * k)
        midx = pd.MultiIndex.from_arrays([qdf["symbol"], lag_dates])
        lag_val = val_lookup.reindex(midx).values.astype(float)
        lag_rd = rd_lookup.reindex(midx).values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pit_ok = np.array([
                (lrd is not pd.NaT
                 and not pd.isnull(lrd)
                 and lrd <= crd)
                for lrd, crd in zip(lag_rd, cur_rd)
            ])

        window_vals[:, T - k] = np.where(pit_ok, lag_val, np.nan)

    valid_count = np.sum(~np.isnan(window_vals), axis=1)
    x_oldest = window_vals[:, 0]    # x_{t-T}
    x_current = window_vals[:, T]   # x_t

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std_x = np.nanstd(window_vals, axis=1, ddof=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_robust = (x_current - x_oldest) / std_x

    invalid = (
        (valid_count < MIN_VALID_ROBUST)
        | np.isnan(x_oldest)
        | np.isnan(x_current)
        | (std_x == 0)
        | np.isnan(std_x)
    )
    g_robust = np.where(invalid, np.nan, g_robust)

    col_name = f"robust_T{T}"
    qdf = qdf.copy()
    qdf[col_name] = g_robust

    return _build_daily_panel(qdf, col_name, trading_dates, end_date)


# ---------------------------------------------------------------------------
# Factor class
# ---------------------------------------------------------------------------

class TA_GROWTH_COMP(FundamentalFactorCalculator):
    """
    Total Assets Growth Composite factor.

    Combines 2 sub-factors for total assets (q_bs_ta_t) growth:
      SURPRISE_GROWTH (T=8), ROBUST_GROWTH (T=20).

    Each sub-factor is independently winsorized and z-scored cross-sectionally,
    then equal-weighted into the composite (>= 1 valid sub-factor required).
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
            "ta_field": TA_FIELD,
            "sub_factors": ["surprise_T8", "robust_T20"],
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute TA_GROWTH_COMP daily panel (N_stocks x T_days).

        Steps:
          1. Load raw fundamental data.
          2. Build deduplicated quarterly panel for TA_FIELD (q_bs_ta_t).
          3. Compute 2 sub-factor daily panels:
             - SURPRISE_GROWTH (T=8)
             - ROBUST_GROWTH (T=20)
          4. Per sub-factor: cross-section winsorize + z-score.
          5. Equal-weight average (need >= MIN_VALID_SUBS valid per cell).
          6. Cross-section rank normalize to [0, 1].
          7. Filter to A-share mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        end_date = fundamental_data.end_date

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("TA_GROWTH_COMP: no raw fundamental data available")

        raw = fundamental_data._raw_data

        # Step 2: Build deduplicated quarterly panel (with positivity filter)
        qdf_ta = _build_quarterly_panel(raw, TA_FIELD)

        if qdf_ta.empty:
            raise ValueError(f"TA_GROWTH_COMP: no valid {TA_FIELD} data after filtering")

        # Step 3: Compute sub-factor daily panels
        sub_panels = []
        sub_names = []

        panel = _compute_surprise_growth_panel(qdf_ta, T_SURPRISE, trading_dates, end_date)
        if panel is not None and not panel.empty:
            sub_panels.append(panel)
            sub_names.append(f"surprise_T{T_SURPRISE}")

        panel = _compute_robust_growth_panel(qdf_ta, T_ROBUST, trading_dates, end_date)
        if panel is not None and not panel.empty:
            sub_panels.append(panel)
            sub_names.append(f"robust_T{T_ROBUST}")

        if len(sub_panels) < MIN_VALID_SUBS:
            raise ValueError(
                f"TA_GROWTH_COMP: only {len(sub_panels)} valid sub-panels "
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
            raise ValueError("TA_GROWTH_COMP: no mainboard stocks after filtering")

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(f"TA_GROWTH_COMP NaN ratio is high: {nan_ratio:.1%}")

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
# Smoke test  (run: python ta_growth_comp.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("TA_GROWTH_COMP factor smoke test")
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

    print(f"\n[Step 2] Compute TA_GROWTH_COMP factor")
    calculator = TA_GROWTH_COMP()
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
