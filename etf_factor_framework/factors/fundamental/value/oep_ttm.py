"""
OEP_TTM factor (Owner Earnings to Price, TTM)

Formula:
    owner_earnings_ttm = np_ttm + rade_ttm
    OEP_TTM = owner_earnings_ttm / mc

Where:
    np_ttm   = mc / pe_ttm          (net profit TTM, 亿 CNY, from lixinger.fundamental)
    rade_ttm = rolling-4Q sum of q_ps_rade_c / 1e8
               (R&D expense TTM, 亿 CNY, from lixinger.financial_statements)
    mc       = total market cap     (亿 CNY, from lixinger.fundamental)

Data fields:
  - pe_ttm          : lixinger.fundamental (daily)
  - mc              : lixinger.fundamental (daily, 亿 CNY)
  - q_ps_rade_c     : lixinger.financial_statements (quarterly single-quarter, CNY)

Factor direction: positive (higher OEP = cheaper relative to owner earnings = better)
Factor category: value - improved value

Notes:
  - Simplified version: omits impairment provisions and deferred tax (fields not in DB).
  - When q_ps_rade_c is NaN for a stock-date, rade_ttm = 0 (falls back to EP_TTM).
  - Stocks with |pe_ttm| < 1e-6 yield NaN (near-zero net profit).
  - mc = 0 or NaN yields NaN.
  - Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only.
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

FACTOR_NAME = "OEP_TTM"
FACTOR_DIRECTION = 1  # positive: higher OEP (cheaper valuation) is better
MIN_ABS_PE = 1e-6     # |pe_ttm| below this threshold => near-zero earnings => NaN


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _compute_rade_ttm_panel(fundamental_data: FundamentalData, trading_dates):
    """
    Compute R&D expense TTM panel as a daily N x T array.

    Uses q_ps_rade_c (single-quarter, CNY) from financial_statements.
    TTM = rolling sum of last 4 single-quarter values, keyed on report_date.
    Forward-filled to trading_dates.

    Returns:
        (values: np.ndarray (N, T) float64, symbols: np.ndarray (N,))
        Values are in CNY (not converted to 亿). Returns (empty, empty) on failure.
    """
    fundamental_data._load_raw_data()
    raw_df = fundamental_data._raw_data

    if raw_df is None or raw_df.empty or "q_ps_rade_c" not in raw_df.columns:
        return np.empty((0, len(trading_dates)), dtype=np.float64), np.array([])

    df = raw_df[["symbol", "date", "report_date", "q_ps_rade_c"]].copy()
    df = df.dropna(subset=["report_date"])

    # Keep only rows with non-null rade, then deduplicate by (symbol, date)
    # keeping the latest report_date (most recent disclosure for that quarter)
    df_rade = df.dropna(subset=["q_ps_rade_c"]).copy()
    if df_rade.empty:
        return np.empty((0, len(trading_dates)), dtype=np.float64), np.array([])

    df_rade = df_rade.sort_values(["symbol", "date", "report_date"])
    df_rade = df_rade.groupby(["symbol", "date"], as_index=False).last()

    # Rolling 4-quarter TTM sum per stock (sorted by quarter-end date)
    df_rade = df_rade.sort_values(["symbol", "date"])
    df_rade["rade_ttm"] = df_rade.groupby("symbol")["q_ps_rade_c"].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )
    df_rade = df_rade.dropna(subset=["rade_ttm"])

    # Filter out future data leakage
    df_rade = df_rade[df_rade["report_date"] <= fundamental_data.end_date]
    if df_rade.empty:
        return np.empty((0, len(trading_dates)), dtype=np.float64), np.array([])

    # Pivot on report_date (signal date) and forward-fill to trading days
    pivot = df_rade.pivot_table(
        index="symbol",
        columns="report_date",
        values="rade_ttm",
        aggfunc="last",
    )

    if trading_dates.tz is not None:
        trading_dates = trading_dates.tz_localize(None)

    all_dates = pivot.columns.union(trading_dates).sort_values()
    pivot = pivot.reindex(columns=all_dates)
    panel = pivot.ffill(axis=1).reindex(columns=trading_dates)
    values = panel.values.astype(np.float64)
    symbols_arr = np.array(panel.index.tolist())

    return values, symbols_arr


def _align_panels(
    pe_values, symbols_pe,
    mc_values, symbols_mc,
    rade_values, symbols_rade,
    dates,
):
    """
    Align three panels onto the common symbol set of pe and mc panels.

    pe and mc must have the same dates (T). rade may have a different symbol set.

    Returns:
        (pe_aligned, mc_aligned, rade_aligned, symbols_common, dates)
        rade_aligned: (N, T) with NaN where a stock has no rade data
    """
    # Common symbols for pe and mc
    pe_sym_set = set(symbols_pe)
    mc_sym_set = set(symbols_mc)
    common_syms = sorted(pe_sym_set & mc_sym_set)

    pe_idx = {s: i for i, s in enumerate(symbols_pe)}
    mc_idx = {s: i for i, s in enumerate(symbols_mc)}
    rows_pe = [pe_idx[s] for s in common_syms]
    rows_mc = [mc_idx[s] for s in common_syms]

    pe_aligned = pe_values[rows_pe]
    mc_aligned = mc_values[rows_mc]
    symbols_common = np.array(common_syms)
    T = len(dates)
    N = len(common_syms)

    # Build rade_aligned: NaN for stocks not in rade panel
    rade_aligned = np.full((N, T), np.nan, dtype=np.float64)
    if rade_values.size > 0 and len(symbols_rade) > 0:
        rade_idx = {s: i for i, s in enumerate(symbols_rade)}
        for i, sym in enumerate(common_syms):
            if sym in rade_idx:
                rade_aligned[i] = rade_values[rade_idx[sym]]

    return pe_aligned, mc_aligned, rade_aligned, symbols_common, dates


class OEP_TTM(FundamentalFactorCalculator):
    """
    Owner Earnings to Price (TTM) factor.

    Simplified formula:
        np_ttm        = mc / pe_ttm            (net profit TTM, 亿 CNY)
        rade_ttm      = rolling-4Q q_ps_rade_c / 1e8  (R&D TTM, 亿 CNY; 0 if unavailable)
        owner_earn    = np_ttm + rade_ttm
        OEP_TTM       = owner_earn / mc

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
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

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Calculate OEP_TTM daily panel.

        Returns:
            FactorData: N stocks x T days, values = OEP_TTM (float64)
        """
        # --- Load valuation panels ---
        pe_values, symbols_pe, dates = fundamental_data.get_valuation_panel("pe_ttm")
        mc_values, symbols_mc, _ = fundamental_data.get_valuation_panel("mc")

        # --- Apply mainboard filter to pe and mc independently ---
        mask_pe = np.array([_is_mainboard(s) for s in symbols_pe])
        pe_values = pe_values[mask_pe]
        symbols_pe = symbols_pe[mask_pe]

        mask_mc = np.array([_is_mainboard(s) for s in symbols_mc])
        mc_values = mc_values[mask_mc]
        symbols_mc = symbols_mc[mask_mc]

        # --- Compute R&D TTM panel (quarterly data, forward-filled) ---
        trading_dates = fundamental_data.trading_dates
        rade_values, symbols_rade = _compute_rade_ttm_panel(fundamental_data, trading_dates)

        # Mainboard filter for rade panel
        if len(symbols_rade) > 0:
            mask_rade = np.array([_is_mainboard(s) for s in symbols_rade])
            rade_values = rade_values[mask_rade]
            symbols_rade = symbols_rade[mask_rade]

        # --- Align all panels on common (pe ∩ mc) symbol universe ---
        pe_aligned, mc_aligned, rade_aligned, symbols_common, dates = _align_panels(
            pe_values, symbols_pe,
            mc_values, symbols_mc,
            rade_values, symbols_rade,
            dates,
        )

        # --- Compute np_ttm = mc / pe_ttm ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_ttm = np.where(
                np.abs(pe_aligned) < MIN_ABS_PE,
                np.nan,
                mc_aligned / pe_aligned,
            )
        np_ttm = np.where(np.isinf(np_ttm), np.nan, np_ttm)

        # --- Compute rade_ttm in 亿 CNY (NaN -> 0, falls back to EP_TTM) ---
        rade_ttm_yi = np.where(np.isnan(rade_aligned), 0.0, rade_aligned / 1e8)

        # --- owner_earnings = np_ttm + rade_ttm ---
        owner_earnings = np_ttm + rade_ttm_yi

        # --- OEP_TTM = owner_earnings / mc ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            oep_values = np.where(
                (mc_aligned == 0) | np.isnan(mc_aligned),
                np.nan,
                owner_earnings / mc_aligned,
            )
        oep_values = np.where(np.isinf(oep_values), np.nan, oep_values)
        oep_values = oep_values.astype(np.float64)

        nan_ratio = np.isnan(oep_values).mean() if oep_values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"OEP_TTM NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=oep_values,
            symbols=symbols_common,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python oep_ttm.py)
# Uses full market with short date range to ensure enough cross-section.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("OEP_TTM factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute OEP_TTM factor")
    calculator = OEP_TTM()
    result = calculator.calculate(fd)

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range   : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # ---- Smoke-test assertions ----------------------------------------
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, \
        f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"
    print(f"NaN ratio    : {nan_ratio:.1%}")

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid    : {len(valid_cs)}")
        print(f"  mean       : {valid_cs.mean():.6f}")
        print(f"  std        : {valid_cs.std():.6f}")
        print(f"  min        : {valid_cs.min():.4f}")
        print(f"  max        : {valid_cs.max():.4f}")
        print(f"  median     : {np.median(valid_cs):.6f}")

    # Sanity: check known stocks have plausible OEP_TTM values
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]
    fd_test = FundamentalData(
        start_date="2024-01-01",
        end_date="2024-12-31",
        stock_codes=TEST_CODES,
    )
    result_test = calculator.calculate(fd_test)
    print(f"\nSample values (5 test stocks, last 5 dates):")
    for i, sym in enumerate(result_test.symbols):
        row = result_test.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {np.round(last5, 4)}")

    print(f"\n[PASS] Smoke test passed: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    fd_leak = FundamentalData(start_date="2016-01-01", end_date="2025-12-31", stock_codes=None)
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
