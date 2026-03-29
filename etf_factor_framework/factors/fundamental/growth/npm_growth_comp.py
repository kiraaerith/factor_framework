"""
NPM_GROWTH_COMP factor (Net Profit Margin Growth Composite)

Synthesizes 8 sub-factor components across multiple time windows and growth
metrics to capture multi-dimensional NPM (net profit margin) growth dynamics.

Components:
  1. PCT_GROWTH  T=2   (semi-annual NPM change)
  2. PCT_GROWTH  T=4   (annual NPM change / YoY)
  3. PCT_GROWTH  T=20  (5-year NPM change)
  4. SURPRISE_GROWTH T=8 (NPM vs 2-year history)
  5. ACCEL T=4         (1-year y2y acceleration)
  6. ACCEL T=20        (5-year y2y acceleration)
  7. ROBUST_GROWTH T=4  (1-year risk-adjusted growth)
  8. ROBUST_GROWTH T=20 (5-year risk-adjusted growth)

Data source:
  lixinger.financial_statements
    q_ps_npatoshopc_c  (single-quarter attributable NI)
    q_ps_oi_c          (single-quarter operating revenue)

  Derived: npm_t = q_ps_npatoshopc_c / q_ps_oi_c

Factor direction: positive (higher composite NPM growth is better)
Factor category: growth - pricing power / bargaining power dimension

Notes:
  - npm_t can be negative (loss-making firms); PCT_GROWTH denominator = |npm_{t-T}|.
  - PCT_GROWTH: near-zero denominator (|npm_{t-T}| < 0.001) -> NaN.
  - ACCEL uses y2y growth rate: npm_y2y_t = (npm - lag4) / |lag4|,
    winsorized at +-5; |y2y_{t-T}| < 0.01 -> NaN.
  - SURPRISE_GROWTH: requires >= 4 valid historical periods; std=0 -> NaN.
  - ROBUST_GROWTH T=4: requires 5 valid pts; T=20: requires 21 pts; std=0 -> NaN.
  - All lags are PIT-safe: lag report_date <= current report_date.
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

FACTOR_NAME = "NPM_GROWTH_COMP"
FACTOR_DIRECTION = 1  # positive: higher NPM growth is better

# Field names
NI_FIELD  = "q_ps_npatoshopc_c"
REV_FIELD = "q_ps_oi_c"

# Tuning constants
PCT_MIN_ABS_DENOM   = 0.001   # PCT_GROWTH min |npm_{t-T}| (~0.1%)
PCT_WINSORIZE_CAP   = 10.0    # PCT_GROWTH hard cap at +-10x
ACCEL_Y2Y_WINSOR_LO = -5.0   # y2y growth winsorize lower bound
ACCEL_Y2Y_WINSOR_HI =  5.0   # y2y growth winsorize upper bound
ACCEL_MIN_DENOM     =  0.01  # min |y2y_{t-T}| for ACCEL
MIN_VALID_COMPS     =  4     # min valid sub-factors for composite


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


def _build_daily_panel(
    raw: pd.DataFrame,
    col: str,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Build a daily forward-filled panel (symbol x trading_dates) from
    quarterly sub-factor values, using report_date as the availability date.
    """
    df = raw[["symbol", "report_date", col]].dropna(subset=[col]).copy()
    df = df[df["report_date"] <= end_date]
    if df.empty:
        return None
    pivot = df.pivot_table(
        index="symbol", columns="report_date", values=col, aggfunc="last"
    )
    all_dates = pivot.columns.union(trading_dates).sort_values()
    pivot = pivot.reindex(columns=all_dates)
    panel = pivot.ffill(axis=1).reindex(columns=trading_dates)
    return panel


# ---------------------------------------------------------------------------
# Sub-factor helpers (operate on pd.Series from the quarterly DataFrame)
# ---------------------------------------------------------------------------

def _pct_growth(x_t: pd.Series, x_lag: pd.Series) -> pd.Series:
    """PCT_GROWTH = (x_t - x_lag) / |x_lag|; hard cap at +-PCT_WINSORIZE_CAP."""
    abs_denom = x_lag.abs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (x_t - x_lag) / abs_denom
    result = result.where(abs_denom >= PCT_MIN_ABS_DENOM, other=np.nan)
    result = result.where(~x_t.isna() & ~x_lag.isna(), other=np.nan)
    result = result.clip(-PCT_WINSORIZE_CAP, PCT_WINSORIZE_CAP)
    return result


def _surprise_growth(x_t: pd.Series,
                     hist_mean: pd.Series,
                     hist_std: pd.Series) -> pd.Series:
    """SURPRISE_GROWTH = (x_t - hist_mean) / hist_std; std=0 -> NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (x_t - hist_mean) / hist_std
    result = result.where(hist_std > 0, other=np.nan)
    result = result.where(
        ~x_t.isna() & ~hist_mean.isna() & ~hist_std.isna(), other=np.nan
    )
    return result


def _accel_y2y(y2y_t: pd.Series, y2y_lag_T: pd.Series) -> pd.Series:
    """
    ACCEL = (y2y_t - y2y_{t-T}) / |y2y_{t-T}|
    Both y2y inputs should already be winsorized.
    """
    abs_denom = y2y_lag_T.abs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (y2y_t - y2y_lag_T) / abs_denom
    result = result.where(abs_denom >= ACCEL_MIN_DENOM, other=np.nan)
    result = result.where(~y2y_t.isna() & ~y2y_lag_T.isna(), other=np.nan)
    return result


def _robust_growth(x_t: pd.Series,
                   x_lag_T: pd.Series,
                   std_window: pd.Series) -> pd.Series:
    """ROBUST_GROWTH = (x_t - x_{t-T}) / std_window; std=0 -> NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (x_t - x_lag_T) / std_window
    result = result.where(std_window > 0, other=np.nan)
    result = result.where(
        ~x_t.isna() & ~x_lag_T.isna() & ~std_window.isna(), other=np.nan
    )
    return result


# ---------------------------------------------------------------------------
# Normalization helpers (vectorized, operate on N x T arrays)
# ---------------------------------------------------------------------------

def _winsorize_3sigma(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional 3-sigma winsorize (vectorized over all T days)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0, keepdims=True)
        std  = np.nanstd(arr,  axis=0, ddof=1, keepdims=True)
    lo = mean - 3.0 * std
    hi = mean + 3.0 * std
    result = np.clip(arr, lo, hi)
    result[np.isnan(arr)] = np.nan
    return result


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score standardization (vectorized over all T days)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0, keepdims=True)
        std  = np.nanstd(arr,  axis=0, ddof=1, keepdims=True)
    std_safe = np.where(std < 1e-10, np.nan, std)
    return (arr - mean) / std_safe


# ---------------------------------------------------------------------------
# Factor class
# ---------------------------------------------------------------------------

class NPM_GROWTH_COMP(FundamentalFactorCalculator):
    """
    Net Profit Margin Growth Composite factor.

    Combines 8 sub-factors for NPM growth (derived from NI/Rev quarterly):
      PCT_GROWTH (T=2, 4, 20), SURPRISE_GROWTH (T=8),
      ACCEL (T=4, 20) -- y2y-based, ROBUST_GROWTH (T=4, 20).

    Each sub-factor is independently 3sigma-winsorized and z-scored
    cross-sectionally, then equal-weighted into the composite
    (>= MIN_VALID_COMPS valid sub-factors required).
    Output restricted to A-share mainboard (60xxxx / 00xxxx).
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
        Compute NPM_GROWTH_COMP daily panel (N stocks x T days).

        Steps:
          1. Extract q_ps_npatoshopc_c and q_ps_oi_c quarterly rows.
          2. Derive npm = NI / Rev; deduplicate; compute PIT lags 1-24.
          3. Compute 8 sub-factor values per quarterly row.
          4. Map each sub-factor to daily panel via report_date ffill.
          5. Normalize each daily panel (3sigma-clip + z-score).
          6. Equal-weight composite; NaN where < MIN_VALID_COMPS valid.
          7. Restrict to A-share mainboard symbols.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("NPM_GROWTH_COMP: no raw fundamental data available")

        # --- Step 1: Extract quarterly data ---
        needed_cols = ["symbol", "date", "report_date", NI_FIELD, REV_FIELD]
        raw = fundamental_data._raw_data[needed_cols].copy()

        # Drop rows where both fields are NaN
        raw = raw.dropna(subset=[NI_FIELD, REV_FIELD], how="all")
        raw = raw[raw["report_date"] <= fundamental_data.end_date].copy()

        # Sanity filter: fiscal period must not exceed report date
        raw = raw[raw["date"] <= raw["report_date"]].copy()

        if raw.empty:
            raise ValueError(
                "NPM_GROWTH_COMP: no data for NI/Rev fields after filtering"
            )

        # Deduplicate by (symbol, fiscal date): keep earliest published version
        raw = raw.sort_values(["symbol", "date", "report_date"])
        raw = raw.groupby(["symbol", "date"], as_index=False).first()
        raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

        # --- Step 2: Derive npm = NI / Rev ---
        ni_vals  = raw[NI_FIELD].values.astype(float)
        rev_vals = raw[REV_FIELD].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            npm_vals = np.where(
                (rev_vals == 0) | np.isnan(rev_vals) | np.isnan(ni_vals),
                np.nan,
                ni_vals / rev_vals,
            )
        raw["npm"] = npm_vals

        # Build (symbol, fiscal_date) -> (npm, report_date) lookup
        raw_idx   = raw.set_index(["symbol", "date"])
        npm_lookup = raw_idx["npm"]
        rd_lookup  = raw_idx["report_date"]

        # Compute PIT lags 1-24 quarters
        for lag in range(1, 25):
            lag_dates = raw["date"] - pd.DateOffset(months=3 * lag)
            midx = pd.MultiIndex.from_arrays([raw["symbol"], lag_dates])
            lag_npm = npm_lookup.reindex(midx).values.astype(float)
            lag_rd  = rd_lookup.reindex(midx).values
            cur_rd  = raw["report_date"].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pit_ok = np.array([
                    (lrd is not pd.NaT
                     and not pd.isnull(lrd)
                     and lrd <= crd)
                    for lrd, crd in zip(lag_rd, cur_rd)
                ])
            raw[f"lag{lag}"] = np.where(pit_ok, lag_npm, np.nan)

        # Compute rolling stats for SURPRISE_GROWTH (lag1..lag8, min 4 valid)
        surp_arr = raw[[f"lag{i}" for i in range(1, 9)]].values
        n_valid8 = (~np.isnan(surp_arr)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw["surp_mean"] = np.where(
                n_valid8 >= 4, np.nanmean(surp_arr, axis=1), np.nan
            )
            raw["surp_std"] = np.where(
                n_valid8 >= 4, np.nanstd(surp_arr, axis=1, ddof=1), np.nan
            )

        # Compute rolling std for ROBUST_GROWTH T=4 (npm + lag1..lag4, min 5)
        r4_arr  = raw[["npm"] + [f"lag{i}" for i in range(1, 5)]].values
        n_valid5 = (~np.isnan(r4_arr)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw["robust4_std"] = np.where(
                n_valid5 >= 5, np.nanstd(r4_arr, axis=1, ddof=1), np.nan
            )

        # Compute rolling std for ROBUST_GROWTH T=20 (npm + lag1..lag20, min 21)
        r20_arr  = raw[["npm"] + [f"lag{i}" for i in range(1, 21)]].values
        n_valid21 = (~np.isnan(r20_arr)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw["robust20_std"] = np.where(
                n_valid21 >= 21, np.nanstd(r20_arr, axis=1, ddof=1), np.nan
            )

        # --- Step 3: Compute 8 sub-factor values per quarterly row ---

        # comp1: PCT_GROWTH T=2
        raw["comp1"] = _pct_growth(raw["npm"], raw["lag2"])

        # comp2: PCT_GROWTH T=4
        raw["comp2"] = _pct_growth(raw["npm"], raw["lag4"])

        # comp3: PCT_GROWTH T=20
        raw["comp3"] = _pct_growth(raw["npm"], raw["lag20"])

        # comp4: SURPRISE_GROWTH T=8
        raw["comp4"] = _surprise_growth(
            raw["npm"], raw["surp_mean"], raw["surp_std"]
        )

        # ACCEL: compute y2y growth rates (winsorized at +-5)
        # y2y_t = (npm - lag4) / |lag4|
        y2y_t = _pct_growth(raw["npm"], raw["lag4"]).clip(
            ACCEL_Y2Y_WINSOR_LO, ACCEL_Y2Y_WINSOR_HI
        )
        # y2y_lag4 = (lag4 - lag8) / |lag8|
        y2y_lag4 = _pct_growth(raw["lag4"], raw["lag8"]).clip(
            ACCEL_Y2Y_WINSOR_LO, ACCEL_Y2Y_WINSOR_HI
        )
        # y2y_lag20 = (lag20 - lag24) / |lag24|
        y2y_lag20 = _pct_growth(raw["lag20"], raw["lag24"]).clip(
            ACCEL_Y2Y_WINSOR_LO, ACCEL_Y2Y_WINSOR_HI
        )

        # comp5: ACCEL T=4  => (y2y_t - y2y_lag4) / |y2y_lag4|
        raw["comp5"] = _accel_y2y(y2y_t, y2y_lag4)

        # comp6: ACCEL T=20 => (y2y_t - y2y_lag20) / |y2y_lag20|
        raw["comp6"] = _accel_y2y(y2y_t, y2y_lag20)

        # comp7: ROBUST_GROWTH T=4
        raw["comp7"] = _robust_growth(raw["npm"], raw["lag4"], raw["robust4_std"])

        # comp8: ROBUST_GROWTH T=20
        raw["comp8"] = _robust_growth(raw["npm"], raw["lag20"], raw["robust20_std"])

        # --- Step 4 + 5: Build daily panels and normalize ---
        comp_cols = [f"comp{i}" for i in range(1, 9)]
        normalized = []   # list of (symbol_list, np.ndarray (n, T))

        for col in comp_cols:
            panel_df = _build_daily_panel(
                raw, col, trading_dates, fundamental_data.end_date
            )
            if panel_df is None or panel_df.empty:
                continue

            arr = panel_df.values.astype(np.float64)
            arr = _winsorize_3sigma(arr)
            arr = _zscore(arr)
            normalized.append((panel_df.index.tolist(), arr))

        if not normalized:
            raise ValueError("NPM_GROWTH_COMP: all sub-factor panels are empty")

        # --- Step 6: Align sub-factors and composite ---
        all_symbols = sorted(set().union(*[set(s) for s, _ in normalized]))
        N = len(all_symbols)
        T = len(trading_dates_arr)
        sym_to_i = {s: i for i, s in enumerate(all_symbols)}

        stacked = np.full((len(normalized), N, T), np.nan, dtype=np.float64)
        for k, (syms, arr) in enumerate(normalized):
            for local_i, sym in enumerate(syms):
                gi = sym_to_i.get(sym)
                if gi is not None:
                    stacked[k, gi, :] = arr[local_i, :]

        valid_count = np.sum(~np.isnan(stacked), axis=0)   # (N, T)
        composite   = np.nanmean(stacked, axis=0)           # (N, T)
        composite[valid_count < MIN_VALID_COMPS] = np.nan

        # --- Step 7: Mainboard filter ---
        symbols_arr = np.array(all_symbols)
        mb_mask = np.array([_is_mainboard(s) for s in all_symbols])
        composite   = composite[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        if len(symbols_arr) == 0:
            raise ValueError(
                "NPM_GROWTH_COMP: no mainboard stocks after filtering"
            )

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"NPM_GROWTH_COMP NaN ratio is high: {nan_ratio:.1%}"
            )

        return FactorData(
            values=composite.astype(np.float64),
            symbols=symbols_arr,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ---------------------------------------------------------------------------
# Smoke test (run: python npm_growth_comp.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("NPM_GROWTH_COMP factor smoke test")
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

    print(f"\n[Step 2] Compute NPM_GROWTH_COMP factor")
    calculator = NPM_GROWTH_COMP()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: "
          f"{pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio: {nan_ratio:.1%}")

    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {np.round(last5, 4)}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid = last_cs[~np.isnan(last_cs)]
    if len(valid):
        print(f"  N valid: {len(valid)}")
        print(f"  mean:    {valid.mean():.4f}")
        print(f"  median:  {np.median(valid):.4f}")
        print(f"  min:     {valid.min():.4f}")
        print(f"  max:     {valid.max():.4f}")
    else:
        print("  No valid values in last cross-section")

    print(f"\n[Step 3] Smoke test assertions (section 4.1)")

    assert result.values.ndim == 2, "values must be 2-D"
    print("  [OK] ndim == 2")

    assert result.values.dtype == np.float64, (
        f"expected float64, got {result.values.dtype}"
    )
    print("  [OK] dtype == float64")

    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"  [OK] NaN ratio {nan_ratio:.1%} < 80%")

    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"
    print("  [OK] No inf values")

    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all(
        (result.values == result2.values) | both_nan
    ), "Idempotency failed"
    print("  [OK] Idempotency")

    print(f"\n[PASS] Smoke test: shape={result.values.shape}, "
          f"NaN={nan_ratio:.1%}")

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
