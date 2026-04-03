"""
COMP_ROBUST_GROWTH factor (Composite Robust Growth Rate via Linear Regression)

Computes:  g_robust_comp = b / std(x_{t-T..t})
where b is the OLS slope from regressing T quarterly values on time index [1..T],
and std is the sample standard deviation (ddof=1) of the same T-quarter window.

This is a signal-to-noise ratio (SNR) type growth factor.  A larger positive value
means the financial metric has a strong upward trend AND low volatility.

This is a parametric method-template factor. Default configuration:
  field : q_ps_toi_c  (total operating income, single-quarter)
  T     : 8           (2-year window, 8 quarters)

Data source : lixinger.financial_statements
Factor direction : positive  (higher composite robust growth is better)
Factor category  : growth - composite robust growth method

Notes:
  - Uses single-quarter values (_c suffix for flow items, _t suffix for stock items).
  - Only fiscal periods with report_date <= current report_date are included (PIT-safe).
  - Deduplication: for same (symbol, fiscal date) pairs, the earliest
    published version is used to ensure full/truncated run consistency.
  - Minimum 3 valid data points required for regression; otherwise NaN.
  - Zero std denominator yields NaN to avoid explosion.
  - Output clipped to [-20, 20] to protect against extreme SNR values when std is tiny.
  - Output is restricted to A-share mainboard stocks (60xxxx / 00xxxx).
  - Differs from COMP_GROWTH only in denominator: std(x) instead of mean(x).
    This makes it insensitive to the sign/magnitude of the metric level,
    and more sensitive to trend stability.
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

FACTOR_NAME = "COMP_ROBUST_GROWTH"
FACTOR_DIRECTION = 1  # positive: higher composite robust growth is better

# Clip output to avoid extreme SNR values when std is very small
CLIP_BOUND = 20.0


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
    quarterly factor values, using report_date as the availability date.

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


def _compute_comp_robust_growth_series(
    vals_arr: np.ndarray, rd_arr: np.ndarray, T: int
) -> np.ndarray:
    """
    Compute COMP_ROBUST_GROWTH for one symbol's quarterly series.

    Args:
        vals_arr : np.ndarray (n,) float64 - quarterly field values sorted by fiscal date
        rd_arr   : np.ndarray (n,) datetime64 - corresponding report_dates
        T        : int - lookback window in quarters

    Returns:
        np.ndarray (n,) float64 - g_robust_comp for each quarterly row

    PIT safety note:
        The window is built from PIT-valid quarters only (report_date <= cur_rd),
        then the last T such quarters are used for regression.  This avoids a
        positional-shift bug where a late-disclosure quarter (report_date > cur_rd)
        occupies a slot in a naive positional window, causing full-run and
        truncated-run computations to use different effective data points.
    """
    n = len(vals_arr)
    results = np.full(n, np.nan)

    for i in range(n):
        cur_rd = rd_arr[i]

        # Collect indices of ALL quarters up to i whose report_date <= cur_rd
        pit_ok = rd_arr[:i + 1] <= cur_rd
        pit_indices = np.where(pit_ok)[0]

        # Take the last T PIT-valid quarters as the regression window
        window_indices = pit_indices[-T:] if len(pit_indices) >= T else pit_indices
        x = vals_arr[window_indices].astype(float)

        valid_mask = ~np.isnan(x)
        n_valid = valid_mask.sum()
        if n_valid < 3:
            continue

        t_idx = np.arange(1, len(x) + 1, dtype=float)
        b = np.polyfit(t_idx[valid_mask], x[valid_mask], 1)[0]

        # Use sample std (ddof=1) of valid values in the window
        std_x = np.nanstd(x, ddof=1)

        if np.isnan(b) or np.isnan(std_x) or std_x == 0.0:
            continue

        val = b / std_x
        # Clip to protect against extreme SNR values
        results[i] = np.clip(val, -CLIP_BOUND, CLIP_BOUND)

    return results


class COMP_ROBUST_GROWTH(FundamentalFactorCalculator):
    """
    Composite Robust Growth Rate factor using linear regression over T quarters.

    Computes: g_robust_comp = b / std(x_{t-T..t})
    where b is OLS slope over the T-quarter window, and std is the sample
    standard deviation (ddof=1) of the same window.

    Parameters
    ----------
    field : str
        lixinger.financial_statements field to use.
        Default: 'q_ps_toi_c' (total operating income, single-quarter).
    T : int
        Lookback window in quarters. Default: 8 (2-year trend).
    """

    def __init__(self, field: str = "q_ps_toi_c", T: int = 8):
        self.field = field
        self.T = T

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"direction": FACTOR_DIRECTION, "field": self.field, "T": self.T}

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute COMP_ROBUST_GROWTH daily panel (N stocks x T_days).

        Steps:
          1. Extract quarterly rows for the target field.
          2. Deduplicate: keep earliest report_date per (symbol, fiscal_date).
          3. Apply sanity filter: fiscal date <= report_date.
          4. Per-symbol: run T-quarter rolling linear regression (PIT-safe).
          5. Forward-fill regression results to daily panel via report_date.
          6. Restrict to A-share mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("COMP_ROBUST_GROWTH: no raw fundamental data available")

        # --- Step 1: Extract relevant columns ---
        needed = ["symbol", "date", "report_date", self.field]
        raw = fundamental_data._raw_data[needed].copy()
        raw = raw.dropna(subset=[self.field])
        raw = raw[raw["report_date"] <= fundamental_data.end_date]

        # Sanity: fiscal period end must not be after its own disclosure date
        raw = raw[raw["date"] <= raw["report_date"]].copy()

        if raw.empty:
            raise ValueError(
                f"COMP_ROBUST_GROWTH: no data for field '{self.field}' after filtering"
            )

        # --- Step 2: Deduplicate ---
        # For the same (symbol, fiscal date), keep the EARLIEST published version.
        # Ensures full-run and truncated-run produce identical window values (PIT-safe).
        raw = raw.sort_values(["symbol", "date", "report_date"])
        raw = raw.groupby(["symbol", "date"], as_index=False).first()
        raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

        # --- Step 3: Compute COMP_ROBUST_GROWTH per symbol ---
        comp_robust_growth_col = np.full(len(raw), np.nan)

        for symbol, grp in raw.groupby("symbol", sort=False):
            grp_sorted = grp.sort_values("date")
            idx = grp_sorted.index

            vals_arr = grp_sorted[self.field].values.astype(float)
            rd_arr = grp_sorted["report_date"].values  # datetime64[ns]

            g_arr = _compute_comp_robust_growth_series(vals_arr, rd_arr, self.T)
            comp_robust_growth_col[idx] = g_arr

        raw["comp_robust_growth"] = comp_robust_growth_col

        # --- Step 4: Forward-fill to daily panel via report_date ---
        panel_df = _build_daily_panel(
            raw, "comp_robust_growth", trading_dates, fundamental_data.end_date
        )
        if panel_df is None or panel_df.empty:
            raise ValueError(
                f"COMP_ROBUST_GROWTH: daily panel is empty for field '{self.field}'"
            )

        values = panel_df.values.astype(np.float64)
        symbols_arr = np.array(panel_df.index.tolist())

        # --- Step 5: Mainboard filter ---
        mb_mask = np.array([_is_mainboard(s) for s in symbols_arr])
        values = values[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        if len(symbols_arr) == 0:
            raise ValueError("COMP_ROBUST_GROWTH: no mainboard stocks after filtering")

        return FactorData(
            values=values,
            symbols=symbols_arr,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test  (run: python comp_robust_growth.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("COMP_ROBUST_GROWTH factor smoke test")
    print("  field=q_ps_toi_c  T=8  (revenue 2-year robust growth SNR)")
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

    print(f"\n[Step 2] Compute COMP_ROBUST_GROWTH factor")
    calculator = COMP_ROBUST_GROWTH(field="q_ps_toi_c", T=8)
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

    # ------------------------------------------------------------------
    # Smoke test assertions (Section 4.1)
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Smoke test assertions")

    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"[PASS] shape={result.values.shape}, NaN={nan_ratio:.1%}, no inf, idempotent")

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
