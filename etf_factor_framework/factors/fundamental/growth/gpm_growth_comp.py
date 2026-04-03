"""
GPM_GROWTH_COMP factor (Gross Profit Margin Composite Growth)

Synthesizes 8 sub-factor components across multiple time windows and growth
metrics to capture multi-dimensional GPM (gross profit margin) growth dynamics.

Components:
  1. PCT_GROWTH  T=2   (semi-annual GPM change)
  2. PCT_GROWTH  T=4   (annual GPM change / YoY)
  3. PCT_GROWTH  T=20  (5-year GPM change)
  4. SURPRISE_GROWTH T=8 (GPM vs 2-year history)
  5. ACCEL T=4         (1-year absolute-change acceleration)
  6. ACCEL T=20        (5-year absolute-change acceleration)
  7. ROBUST_GROWTH T=4  (1-year risk-adjusted growth)
  8. ROBUST_GROWTH T=20 (5-year risk-adjusted growth)

Data source:
  lixinger.financial_statements.q_ps_gp_m_t  (TTM gross profit margin, quarterly)

Factor direction: positive (higher composite GPM growth is better)
Factor category: growth - pricing power / bargaining power dimension

Notes:
  - GPM is a ratio; absolute difference (pp change) is used in ACCEL.
  - Pre-winsorize raw GPM at [-0.5, 1.0] before all sub-factor computation.
  - PCT_GROWTH denominator uses |GPM_{t-T}|; near-zero (<0.5%) -> NaN.
  - ACCEL uses absolute GPM change (pp), winsorized at +-0.3 (30pp).
  - Each sub-factor is independently normalized (3sigma-clip + z-score)
    before equal-weight compositing (min 4 valid sub-factors required).
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

FACTOR_NAME = "GPM_GROWTH_COMP"
FACTOR_DIRECTION = 1  # positive: higher GPM growth is better

# Tuning constants
GPM_WINSORIZE_LO   = -0.5   # pre-winsorize raw GPM lower bound
GPM_WINSORIZE_HI   =  1.0   # pre-winsorize raw GPM upper bound
ACCEL_DELTA_CLIP   =  0.3   # ACCEL delta winsorize at +-30pp
ACCEL_MIN_DENOM    =  0.003 # ACCEL min |delta_{t-T}| (0.3pp)
PCT_MIN_ABS_DENOM  =  0.005 # PCT_GROWTH min |GPM_{t-T}| (0.5%)
PCT_WINSORIZE_CAP  = 10.0   # PCT_GROWTH hard cap at +-10
MIN_VALID_COMPS    =  4     # min valid sub-factors for composite


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class GPM_GROWTH_COMP(FundamentalFactorCalculator):
    """
    Gross Profit Margin Composite Growth factor.

    Compute 8 sub-factors in the quarterly domain (using `q_ps_gp_m_t` lags),
    forward-fill each to daily frequency via report_date, normalize each
    sub-factor independently, then equal-weight composite.
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute GPM_GROWTH_COMP daily panel (N stocks x T days).

        Steps:
          1. Extract q_ps_gp_m_t quarterly rows from raw data.
          2. Pre-winsorize and compute all quarterly lags + rolling stats.
          3. Compute 8 sub-factor values per stock per quarterly report.
          4. Map each sub-factor to a daily panel via report_date ffill.
          5. Normalize each daily panel (3sigma-clip + z-score).
          6. Equal-weight composite; NaN where < MIN_VALID_COMPS valid.
          7. Restrict to A-share mainboard symbols.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("GPM_GROWTH_COMP: no raw fundamental data available")

        # --- Step 1: Extract quarterly data ---
        raw = fundamental_data._raw_data[
            ["symbol", "date", "report_date", "q_ps_gp_m_t"]
        ].copy()

        raw = raw.dropna(subset=["q_ps_gp_m_t"])
        raw = raw[raw["report_date"] <= fundamental_data.end_date].copy()

        # Sanity filter: a quarterly report cannot be filed before the fiscal
        # period ends. Rows where fiscal_date > report_date are data-quality
        # errors.
        raw = raw[raw["date"] <= raw["report_date"]].copy()

        if raw.empty:
            raise ValueError("GPM_GROWTH_COMP: no q_ps_gp_m_t data after filtering")

        # Deduplicate by (symbol, fiscal date): keep the EARLIEST published
        # version of each fiscal quarter (PIT-safe).
        raw = raw.sort_values(["symbol", "date", "report_date"])
        raw = raw.groupby(["symbol", "date"], as_index=False).first()

        # Sort by (symbol, fiscal date) for shift alignment
        raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

        # --- Step 2: Pre-winsorize and compute PIT date-based lags ---
        raw["gpm"] = raw["q_ps_gp_m_t"].clip(GPM_WINSORIZE_LO, GPM_WINSORIZE_HI)

        # Build (symbol, fiscal_date) -> (gpm, report_date) lookup.
        # PIT constraint: a lag is valid only if the lag quarter's report_date
        # <= the current row's report_date.
        raw_idx = raw.set_index(["symbol", "date"])
        gpm_lookup = raw_idx["gpm"]
        rd_lookup  = raw_idx["report_date"]

        # Compute lags 1-24 (quarter-based) with PIT filter
        for lag in range(1, 25):
            lag_dates = raw["date"] - pd.DateOffset(months=3 * lag)
            midx = pd.MultiIndex.from_arrays([raw["symbol"], lag_dates])
            lag_gpm = gpm_lookup.reindex(midx).values.astype(float)
            lag_rd  = rd_lookup.reindex(midx).values
            cur_rd  = raw["report_date"].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pit_ok = np.array([
                    (lrd is not pd.NaT and not pd.isnull(lrd) and lrd <= crd)
                    for lrd, crd in zip(lag_rd, cur_rd)
                ])
            raw[f"lag{lag}"] = np.where(pit_ok, lag_gpm, np.nan)

        # Compute rolling stats from PIT-filtered lag arrays
        # surp_mean / surp_std: lag1..lag8 (previous 8 quarters, min 4)
        surp_arr = raw[[f"lag{i}" for i in range(1, 9)]].values
        n_valid8 = (~np.isnan(surp_arr)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw["surp_mean"] = np.where(n_valid8 >= 4,
                                        np.nanmean(surp_arr, axis=1), np.nan)
            raw["surp_std"]  = np.where(n_valid8 >= 4,
                                        np.nanstd(surp_arr, axis=1, ddof=1), np.nan)

        # robust4_std: gpm + lag1..lag4 (5 quarters, min 3)
        r4_arr = raw[["gpm"] + [f"lag{i}" for i in range(1, 5)]].values
        n_valid5 = (~np.isnan(r4_arr)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw["robust4_std"] = np.where(n_valid5 >= 3,
                                          np.nanstd(r4_arr, axis=1, ddof=1), np.nan)

        # robust20_std: gpm + lag1..lag20 (21 quarters, min 11)
        r20_arr = raw[["gpm"] + [f"lag{i}" for i in range(1, 21)]].values
        n_valid21 = (~np.isnan(r20_arr)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw["robust20_std"] = np.where(n_valid21 >= 11,
                                           np.nanstd(r20_arr, axis=1, ddof=1), np.nan)

        # --- Step 3: Compute 8 sub-factor values per quarterly row ---
        # comp1: PCT_GROWTH T=2
        raw["comp1"] = _pct_growth(raw["gpm"], raw["lag2"])

        # comp2: PCT_GROWTH T=4
        raw["comp2"] = _pct_growth(raw["gpm"], raw["lag4"])

        # comp3: PCT_GROWTH T=20
        raw["comp3"] = _pct_growth(raw["gpm"], raw["lag20"])

        # comp4: SURPRISE_GROWTH T=8
        raw["comp4"] = _surprise_growth(raw["gpm"], raw["surp_mean"], raw["surp_std"])

        # comp5: ACCEL T=4
        #   delta_t     = gpm - lag4
        #   delta_{t-4} = lag4 - lag8
        raw["comp5"] = _accel(raw["gpm"], raw["lag4"], raw["lag4"], raw["lag8"])

        # comp6: ACCEL T=20
        #   delta_t      = gpm - lag4
        #   delta_{t-20} = lag20 - lag24
        raw["comp6"] = _accel(raw["gpm"], raw["lag4"], raw["lag20"], raw["lag24"])

        # comp7: ROBUST_GROWTH T=4
        raw["comp7"] = _robust_growth(raw["gpm"], raw["lag4"], raw["robust4_std"])

        # comp8: ROBUST_GROWTH T=20
        raw["comp8"] = _robust_growth(raw["gpm"], raw["lag20"], raw["robust20_std"])

        # --- Step 4 + 5: Build daily panels and normalize ---
        comp_cols = [f"comp{i}" for i in range(1, 9)]
        normalized = []   # list of (symbol_list, np.ndarray (n, T))

        for col in comp_cols:
            panel_df = _build_daily_panel(raw, col, trading_dates,
                                          fundamental_data.end_date)
            if panel_df is None or panel_df.empty:
                continue

            arr = panel_df.values.astype(np.float64)

            arr = _winsorize_3sigma(arr)
            arr = _zscore(arr)

            normalized.append((panel_df.index.tolist(), arr))

        if not normalized:
            raise ValueError(
                "GPM_GROWTH_COMP: all sub-factor panels are empty"
            )

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

        valid_count = np.sum(~np.isnan(stacked), axis=0)  # (N, T)
        composite = np.nanmean(stacked, axis=0)            # (N, T)
        composite[valid_count < MIN_VALID_COMPS] = np.nan

        # --- Step 7: Mainboard filter ---
        symbols_arr = np.array(all_symbols)
        mb_mask = np.array([_is_mainboard(s) for s in all_symbols])
        composite = composite[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        return FactorData(
            values=composite.astype(np.float64),
            symbols=symbols_arr,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Sub-factor computation (module-level for clarity)
# ------------------------------------------------------------------

def _pct_growth(x_t: pd.Series, x_lag: pd.Series) -> pd.Series:
    """
    PCT_GROWTH = (x_t - x_{t-T}) / |x_{t-T}|

    NaN when: |x_{t-T}| < PCT_MIN_ABS_DENOM, or either value is NaN.
    Hard cap: clipped to [-PCT_WINSORIZE_CAP, +PCT_WINSORIZE_CAP].
    """
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
    """
    SURPRISE_GROWTH = (x_t - mean(hist)) / std(hist)

    NaN when: hist_std == 0, or any input is NaN, or hist has < 4 valid pts
    (min_periods=4 is enforced upstream in the rolling computation).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (x_t - hist_mean) / hist_std
    result = result.where(hist_std > 0, other=np.nan)
    result = result.where(
        ~x_t.isna() & ~hist_mean.isna() & ~hist_std.isna(), other=np.nan
    )
    return result


def _accel(gpm_t: pd.Series,
           gpm_lag4: pd.Series,
           gpm_lag_T: pd.Series,
           gpm_lag_T4: pd.Series) -> pd.Series:
    """
    ACCEL = (delta_t - delta_{t-T}) / |delta_{t-T}|
      where delta_t     = gpm_t - gpm_{t-4}     (current YoY absolute change)
            delta_{t-T} = gpm_{t-T} - gpm_{t-T-4}

    Winsorize deltas at +-ACCEL_DELTA_CLIP before division.
    NaN when: |delta_{t-T}| < ACCEL_MIN_DENOM, or any node is NaN.
    """
    delta_t  = (gpm_t    - gpm_lag4 ).clip(-ACCEL_DELTA_CLIP, ACCEL_DELTA_CLIP)
    delta_tm = (gpm_lag_T - gpm_lag_T4).clip(-ACCEL_DELTA_CLIP, ACCEL_DELTA_CLIP)

    abs_denom = delta_tm.abs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (delta_t - delta_tm) / abs_denom
    result = result.where(abs_denom >= ACCEL_MIN_DENOM, other=np.nan)
    all_valid = (
        ~gpm_t.isna() & ~gpm_lag4.isna() &
        ~gpm_lag_T.isna() & ~gpm_lag_T4.isna()
    )
    result = result.where(all_valid, other=np.nan)
    return result


def _robust_growth(x_t: pd.Series,
                   x_lag_T: pd.Series,
                   std_window: pd.Series) -> pd.Series:
    """
    ROBUST_GROWTH = (x_t - x_{t-T}) / std(window[t-T..t])

    NaN when: std == 0, either endpoint is NaN, or window has < min_periods
    valid pts (min_periods is enforced upstream in the rolling computation).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (x_t - x_lag_T) / std_window
    result = result.where(std_window > 0, other=np.nan)
    result = result.where(
        ~x_t.isna() & ~x_lag_T.isna() & ~std_window.isna(), other=np.nan
    )
    return result


# ------------------------------------------------------------------
# Daily panel construction
# ------------------------------------------------------------------

def _build_daily_panel(
    raw: pd.DataFrame,
    col: str,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Build a daily forward-filled panel (symbol x trading_dates) from
    quarterly sub-factor values, using report_date as the availability date.

    Returns None if no valid data.
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


# ------------------------------------------------------------------
# Normalization helpers (vectorized, operate on N x T arrays)
# ------------------------------------------------------------------

def _industry_fill(arr: np.ndarray,
                   symbols: np.ndarray,
                   industry_map: dict) -> np.ndarray:
    """Fill cross-sectional NaN with industry median per trading day."""
    if not industry_map:
        return arr

    result = arr.copy()
    industries = np.array([industry_map.get(s) for s in symbols], dtype=object)
    valid_mask = np.array([v is not None for v in industries])

    if not valid_mask.any():
        return result

    unique_inds = np.unique(industries[valid_mask].astype(str))

    for ind in unique_inds:
        mask = (industries == ind)
        group = arr[mask, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(group, axis=0)
        nan_in_group = np.isnan(group)
        filled = np.where(nan_in_group, med, group)
        filled = np.where(np.isnan(med), group, filled)
        result[mask, :] = filled

    return result


def _winsorize_3sigma(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional 3-sigma winsorize (vectorized over all T days)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0, keepdims=True)
        std  = np.nanstd(arr,  axis=0, ddof=1, keepdims=True)
    lo = mean - 3.0 * std
    hi = mean + 3.0 * std
    result = np.clip(arr, lo, hi)
    result[np.isnan(arr)] = np.nan  # preserve original NaNs
    return result


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score standardization (vectorized over all T days)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0, keepdims=True)
        std  = np.nanstd(arr,  axis=0, ddof=1, keepdims=True)
    std_safe = np.where(std < 1e-10, np.nan, std)
    return (arr - mean) / std_safe


# ------------------------------------------------------------------
# Smoke test (run: python gpm_growth_comp.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("GPM_GROWTH_COMP factor smoke test")
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

    print(f"\n[Step 2] Compute GPM_GROWTH_COMP factor")
    calculator = GPM_GROWTH_COMP()
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

    # shape
    assert result.values.ndim == 2, "values must be 2-D"
    print("  [OK] ndim == 2")

    # dtype
    assert result.values.dtype == np.float64, (
        f"expected float64, got {result.values.dtype}"
    )
    print("  [OK] dtype == float64")

    # NaN rate
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"  [OK] NaN ratio {nan_ratio:.1%} < 80%")

    # No inf
    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"
    print("  [OK] No inf values")

    # Idempotency
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
