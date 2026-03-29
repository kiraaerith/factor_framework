"""
PCT_GROWTH factor (Percentage Growth Rate)

Computes:  g_pct = (x_t - x_{t-T}) / |x_{t-T}|

This is a parametric method-template factor. Default configuration:
  field : q_ps_toi_c  (total operating income, single-quarter)
  T     : 4           (year-over-year, 4 quarters = 1 year)

Data source : lixinger.financial_statements
Factor direction : positive  (higher growth is better)
Factor category  : growth - percentage growth method

Notes:
  - x_{t-T} is retrieved by shifting fiscal period date back T*3 months.
  - PIT constraint: the lag record's report_date must be <= current
    record's report_date (prevents restatement leakage).
  - Deduplication: for same (symbol, fiscal date) pairs, the earliest
    published version is used to ensure full/truncated run consistency.
  - Denominator guard: |x_{t-T}| < 1.0 (yuan) is treated as zero and
    yields NaN (avoids explosion near-zero denominators).
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

FACTOR_NAME = "PCT_GROWTH"
FACTOR_DIRECTION = 1  # positive: higher growth is better

# Minimum absolute denominator to avoid explosion (1 yuan in lixinger unit)
MIN_ABS_DENOM = 1.0


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


class PCT_GROWTH(FundamentalFactorCalculator):
    """
    Percentage Growth Rate factor.

    Computes: g_pct = (x_t - x_{t-T}) / |x_{t-T}|

    Parameters
    ----------
    field : str
        lixinger.financial_statements field to use.
        Default: 'q_ps_toi_c' (total operating income, single-quarter).
    T : int
        Lookback window in quarters. Default: 4 (year-over-year).
    """

    def __init__(self, field: str = "q_ps_toi_c", T: int = 4):
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

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        Compute PCT_GROWTH daily panel (N stocks x T_days).

        Steps:
          1. Extract quarterly rows for the target field.
          2. Deduplicate and apply PIT / sanity filters.
          3. Compute T-quarter lag via fiscal date shift.
          4. Apply PIT filter to lag values.
          5. Compute g_pct per quarterly row.
          6. Forward-fill to daily panel via report_date.
          7. Restrict to A-share mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("PCT_GROWTH: no raw fundamental data available")

        # --- Step 1: Extract relevant columns ---
        needed = ["symbol", "date", "report_date", self.field]
        raw = fundamental_data._raw_data[needed].copy()
        raw = raw.dropna(subset=[self.field])
        raw = raw[raw["report_date"] <= fundamental_data.end_date]

        # Sanity: fiscal period end must not be after its own disclosure date
        raw = raw[raw["date"] <= raw["report_date"]].copy()

        if raw.empty:
            raise ValueError(
                f"PCT_GROWTH: no data for field '{self.field}' after filtering"
            )

        # --- Step 2: Deduplicate ---
        # For the same (symbol, fiscal date), keep the EARLIEST published version.
        # This ensures full-run and truncated-run produce identical lag values
        # (PIT-safe against late restatements).
        raw = raw.sort_values(["symbol", "date", "report_date"])
        raw = raw.groupby(["symbol", "date"], as_index=False).first()
        raw = raw.sort_values(["symbol", "date"]).reset_index(drop=True)

        # --- Step 3: Compute T-quarter lag using fiscal date shift ---
        raw_idx = raw.set_index(["symbol", "date"])
        val_lookup = raw_idx[self.field]
        rd_lookup = raw_idx["report_date"]

        lag_dates = raw["date"] - pd.DateOffset(months=3 * self.T)
        midx = pd.MultiIndex.from_arrays([raw["symbol"], lag_dates])
        lag_val = val_lookup.reindex(midx).values.astype(float)
        lag_rd = rd_lookup.reindex(midx).values  # object array of Timestamps / NaT

        # --- Step 4: PIT filter on lag ---
        # A lag is valid only if its disclosure date <= current disclosure date.
        cur_rd = raw["report_date"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pit_ok = np.array([
                (lrd is not pd.NaT
                 and not pd.isnull(lrd)
                 and lrd <= crd)
                for lrd, crd in zip(lag_rd, cur_rd)
            ])

        raw["lag_val"] = np.where(pit_ok, lag_val, np.nan)

        # --- Step 5: Compute PCT_GROWTH per quarterly row ---
        x_t = raw[self.field]
        x_lag = raw["lag_val"]
        abs_denom = x_lag.abs()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            growth = (x_t - x_lag) / abs_denom

        # Zero / near-zero denominator -> NaN
        growth = growth.where(abs_denom >= MIN_ABS_DENOM, other=np.nan)
        # Either input NaN -> NaN
        growth = growth.where(~x_t.isna() & ~x_lag.isna(), other=np.nan)

        raw["pct_growth"] = growth

        # --- Step 6: Forward-fill to daily panel via report_date ---
        panel_df = _build_daily_panel(
            raw, "pct_growth", trading_dates, fundamental_data.end_date
        )
        if panel_df is None or panel_df.empty:
            raise ValueError(
                f"PCT_GROWTH: daily panel is empty for field '{self.field}'"
            )

        values = panel_df.values.astype(np.float64)
        symbols_arr = np.array(panel_df.index.tolist())

        # --- Step 7: Mainboard filter ---
        mb_mask = np.array([_is_mainboard(s) for s in symbols_arr])
        values = values[mb_mask]
        symbols_arr = symbols_arr[mb_mask]

        if len(symbols_arr) == 0:
            raise ValueError("PCT_GROWTH: no mainboard stocks after filtering")

        return FactorData(
            values=values,
            symbols=symbols_arr,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test  (run: python pct_growth.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("PCT_GROWTH factor smoke test")
    print("  field=q_ps_toi_c  T=4  (revenue YoY)")
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

    print(f"\n[Step 2] Compute PCT_GROWTH factor")
    calculator = PCT_GROWTH(field="q_ps_toi_c", T=4)
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
