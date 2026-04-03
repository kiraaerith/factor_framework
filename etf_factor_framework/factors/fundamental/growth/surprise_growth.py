"""
SURPRISE_GROWTH factor (Standardized Unexpected Growth / SUE-style)

Computes:  g_surprise = (x_t - mean(x_{t-1}...x_{t-T})) / std(x_{t-1}...x_{t-T})

This factor measures how much the latest financial metric deviates from its
own historical mean, normalized by historical volatility (SUE-style logic).
Positive values indicate the metric exceeded historical expectations.

Default configuration:
  fields : q_ps_toi_c, q_ps_npatoshopc_c, q_ps_op_c, q_cfs_ncffoa_c,
           q_bs_ta_t, q_bs_toe_t
  T      : 8  (two-year window, 8 quarters)

Data source : lixinger.financial_statements
Factor direction : positive  (higher surprise growth is better)
Factor category  : growth - surprise growth method

Notes:
  - Historical lags x_{t-k} are retrieved by shifting the fiscal period date
    back k*3 months (k=1..T). PIT filter: lag record's report_date must be
    <= current record's report_date.
  - Requires at least MIN_HIST_PERIODS (=4) valid historical values; otherwise
    g_surprise = NaN.
  - When hist_std == 0 (flat history), g_surprise = NaN.
  - Six sub-indicators are computed independently, then equal-weighted into
    the composite SURPRISE_GROWTH. Stock must have >=3 valid sub-indicators.
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

FACTOR_NAME = "SURPRISE_GROWTH"
FACTOR_DIRECTION = 1  # positive: higher surprise growth is better

# Default sub-indicators (q_ps_ebit_c excluded due to low coverage ~72%)
DEFAULT_FIELDS = [
    "q_ps_toi_c",          # total operating income, single-quarter
    "q_ps_npatoshopc_c",   # net profit attributable to shareholders, single-quarter
    "q_ps_op_c",           # operating profit, single-quarter
    "q_cfs_ncffoa_c",      # net cash flow from operating activities, single-quarter
    "q_bs_ta_t",           # total assets, point-in-time
    "q_bs_toe_t",          # total owners' equity, point-in-time
]

# Historical window: 8 quarters (two years)
DEFAULT_T = 8

# Minimum valid historical periods required to compute g_surprise
MIN_HIST_PERIODS = 4

# Minimum valid sub-indicators required to compute composite
MIN_VALID_SUBS = 3


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _build_daily_panel(
    df: pd.DataFrame,
    col: str,
    trading_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> "pd.DataFrame | None":
    """
    Build a daily forward-filled panel (symbol x trading_dates) from
    quarterly sub-factor values, using report_date as the availability date.
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


def _compute_surprise_for_field(
    raw: pd.DataFrame,
    field: str,
    T: int,
    end_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> "pd.DataFrame | None":
    """
    Compute g_surprise for a single financial field.

    Returns a daily panel (symbol x trading_dates) or None if insufficient data.
    """
    # Extract rows with valid field values and valid report dates
    needed = ["symbol", "date", "report_date", field]
    sub = raw[needed].copy()
    sub = sub.dropna(subset=[field])
    sub = sub[sub["report_date"] <= end_date]

    # Sanity: fiscal period end must not be after disclosure date
    sub = sub[sub["date"] <= sub["report_date"]].copy()

    if sub.empty:
        return None

    # Deduplicate: for same (symbol, fiscal date), keep earliest published version
    # (PIT-safe against late restatements, consistent between full/truncated runs)
    sub = sub.sort_values(["symbol", "date", "report_date"])
    sub = sub.groupby(["symbol", "date"], as_index=False).first()
    sub = sub.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Build lookup: (symbol, fiscal_date) -> (field_value, report_date)
    sub_idx = sub.set_index(["symbol", "date"])
    val_lookup = sub_idx[field]
    rd_lookup = sub_idx["report_date"]

    # Retrieve T historical lags for each row
    # hist_vals shape: (n_rows, T)
    n_rows = len(sub)
    hist_vals = np.full((n_rows, T), np.nan)

    for k in range(1, T + 1):
        lag_dates = sub["date"] - pd.DateOffset(months=3 * k)
        midx = pd.MultiIndex.from_arrays([sub["symbol"], lag_dates])

        lag_val = val_lookup.reindex(midx).values.astype(float)
        lag_rd = rd_lookup.reindex(midx).values  # Timestamps or NaT

        # PIT filter: lag must have been disclosed before current disclosure
        cur_rd = sub["report_date"].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pit_ok = np.array([
                (lrd is not pd.NaT
                 and not pd.isnull(lrd)
                 and lrd <= crd)
                for lrd, crd in zip(lag_rd, cur_rd)
            ])

        lag_val_pit = np.where(pit_ok, lag_val, np.nan)
        hist_vals[:, k - 1] = lag_val_pit

    # Count valid historical periods per row
    n_valid = np.sum(~np.isnan(hist_vals), axis=1)

    # Compute mean and std over valid historical periods (ignore NaN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hist_mean = np.nanmean(hist_vals, axis=1)
        hist_std = np.nanstd(hist_vals, axis=1, ddof=1)

    x_t = sub[field].values.astype(float)

    # Compute g_surprise = (x_t - mean) / std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_surprise = (x_t - hist_mean) / hist_std

    # Mask out cases with insufficient history or zero std
    mask_insufficient = n_valid < MIN_HIST_PERIODS
    mask_zero_std = hist_std == 0
    mask_nan_xt = np.isnan(x_t)
    invalid = mask_insufficient | mask_zero_std | mask_nan_xt | np.isnan(hist_mean)
    g_surprise = np.where(invalid, np.nan, g_surprise)

    sub["g_surprise"] = g_surprise

    # Forward-fill to daily panel via report_date
    panel = _build_daily_panel(sub, "g_surprise", trading_dates, end_date)
    return panel


class SURPRISE_GROWTH(FundamentalFactorCalculator):
    """
    Standardized Unexpected Growth factor (SUE-style).

    Computes: g_surprise = (x_t - mean(x_{t-1}...x_{t-T})) / std(x_{t-1}...x_{t-T})

    Multiple sub-indicators are computed independently and then
    equal-weighted into the composite factor value.

    Parameters
    ----------
    fields : list of str
        lixinger.financial_statements fields to use as sub-indicators.
        Default: 6 fields (revenue, net profit, op profit, OCF, total assets, equity).
    T : int
        Historical window in quarters. Default: 8 (two-year).
    """

    def __init__(self, fields: list = None, T: int = DEFAULT_T):
        self.fields = fields if fields is not None else DEFAULT_FIELDS
        self.T = T

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
            "fields": self.fields,
            "T": self.T,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute composite SURPRISE_GROWTH daily panel (N stocks x T_days).

        Steps:
          1. Extract raw quarterly data.
          2. For each sub-indicator, compute g_surprise per quarterly row.
             - Retrieve T historical lags via fiscal date shift.
             - Apply PIT filter.
             - Compute mean/std of valid history; derive g_surprise.
          3. Build each sub-indicator's daily panel via report_date ffill.
          4. Align all sub-panels to common symbol universe and trading dates.
          5. Equal-weight across valid sub-indicators (>=MIN_VALID_SUBS required).
          6. Restrict to A-share mainboard stocks.
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        if fundamental_data._raw_data is None or fundamental_data._raw_data.empty:
            raise ValueError("SURPRISE_GROWTH: no raw fundamental data available")

        raw = fundamental_data._raw_data.copy()
        end_date = fundamental_data.end_date

        # Compute sub-panels for each field
        sub_panels = {}
        for field in self.fields:
            if field not in raw.columns:
                warnings.warn(f"SURPRISE_GROWTH: field '{field}' not in data, skipping")
                continue
            panel = _compute_surprise_for_field(raw, field, self.T, end_date, trading_dates)
            if panel is not None and not panel.empty:
                sub_panels[field] = panel

        if len(sub_panels) < MIN_VALID_SUBS:
            raise ValueError(
                f"SURPRISE_GROWTH: only {len(sub_panels)} valid sub-panels, "
                f"need >= {MIN_VALID_SUBS}"
            )

        # Align to union of all symbols, compute equal-weighted composite
        all_symbols = sorted(
            set().union(*[set(p.index.tolist()) for p in sub_panels.values()])
        )

        # Mainboard filter
        mb_mask = np.array([_is_mainboard(s) for s in all_symbols])
        all_symbols = [s for s, ok in zip(all_symbols, mb_mask) if ok]

        if len(all_symbols) == 0:
            raise ValueError("SURPRISE_GROWTH: no mainboard stocks after filtering")

        n_stocks = len(all_symbols)
        n_dates = len(trading_dates)
        symbol_to_idx = {s: i for i, s in enumerate(all_symbols)}

        # Stack sub-panels: shape (n_subs, n_stocks, n_dates)
        n_subs = len(sub_panels)
        stack = np.full((n_subs, n_stocks, n_dates), np.nan)

        for si, (field, panel) in enumerate(sub_panels.items()):
            panel_mb = panel.reindex(index=all_symbols)
            stack[si] = panel_mb.values.astype(np.float64)

        # Count valid (non-NaN) sub-indicators per (stock, date)
        n_valid_subs = np.sum(~np.isnan(stack), axis=0)  # (n_stocks, n_dates)

        # Equal-weighted mean across sub-indicators
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            composite = np.nanmean(stack, axis=0)  # (n_stocks, n_dates)

        # Require at least MIN_VALID_SUBS valid sub-indicators
        composite = np.where(n_valid_subs >= MIN_VALID_SUBS, composite, np.nan)

        symbols_arr = np.array(all_symbols)

        return FactorData(
            values=composite.astype(np.float64),
            symbols=symbols_arr,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test  (run: python surprise_growth.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("SURPRISE_GROWTH factor smoke test")
    print(f"  fields={DEFAULT_FIELDS}")
    print(f"  T={DEFAULT_T}")
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

    print(f"\n[Step 2] Compute SURPRISE_GROWTH factor")
    calculator = SURPRISE_GROWTH()
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
    print(f"\nSmoke test PASSED")

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
