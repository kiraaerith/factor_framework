"""
Net_ROE factor (Single-Quarter Net Return on Equity)

Data fields from lixinger.financial_statements:
  - q_ps_npatoshopc_c : single-quarter net profit attributable to parent shareholders
  - q_bs_toe_t        : total owner's equity (period-end)
  - q_bs_etmsh_t      : minority shareholder equity (period-end), filled 0 if missing

Computation:
  NetAsset_end   = q_bs_toe_t - q_bs_etmsh_t
  NetAsset_begin = previous quarter's NetAsset_end (shift-by-1 per stock)
  AvgNetAsset    = (NetAsset_begin + NetAsset_end) / 2
  Net_ROE        = q_ps_npatoshopc_c / AvgNetAsset

Factor direction: positive (higher ROE is better)
Factor category: profitability

Notes:
  - Uses report_date for forward-fill (no future data leakage).
  - Stocks with AvgNetAsset == 0 or NaN are excluded from each snapshot.
  - Mainboard filter: only SHSE.60xxxx and SZSE.00xxxx.
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

FACTOR_NAME = "Net_ROE"
FACTOR_DIRECTION = 1  # positive: higher ROE is better


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class ROE(FundamentalFactorCalculator):
    """
    Single-Quarter Net Return on Equity (Net_ROE).

    Computes ROE from raw financial fields:
      Net_ROE = q_ps_npatoshopc_c / AvgNetAsset
    where AvgNetAsset is the average of period-beginning and period-ending
    net equity attributable to parent shareholders.

    Forward-fills quarterly values to daily frequency using report_date.
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
        Calculate Net_ROE daily panel.

        Returns:
            FactorData: N stocks × T days, values = Net_ROE (quarterly, ratio)
        """
        # Ensure raw data is loaded
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        raw = fundamental_data._raw_data

        if raw is None or raw.empty:
            empty_vals = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty_vals,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ---------------------------------------------------------------
        # Step 1: Extract needed columns
        # ---------------------------------------------------------------
        needed = ["symbol", "stock_code", "date", "report_date",
                  "q_ps_npatoshopc_c", "q_bs_toe_t", "q_bs_etmsh_t"]
        df = raw[[c for c in needed if c in raw.columns]].copy()

        # ---------------------------------------------------------------
        # Step 2: Mainboard filter (apply early to reduce data size)
        # ---------------------------------------------------------------
        mainboard_mask = np.array([_is_mainboard(s) for s in df["symbol"]])
        df = df[mainboard_mask].copy()

        if df.empty:
            empty_vals = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty_vals,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ---------------------------------------------------------------
        # Step 3: Compute NetAsset_end = total equity - minority equity
        # ---------------------------------------------------------------
        df["q_bs_etmsh_t"] = df["q_bs_etmsh_t"].fillna(0.0)
        df["NetAsset_end"] = df["q_bs_toe_t"] - df["q_bs_etmsh_t"]

        # ---------------------------------------------------------------
        # Step 4: Filter to reports published by end_date.
        # ---------------------------------------------------------------
        df = df[df["report_date"] <= fundamental_data.end_date].copy()

        # ---------------------------------------------------------------
        # Step 5: Point-in-time NetAsset_begin
        # For each quarterly record published at report_date R, look up
        # the most recently published version of the PREVIOUS quarter that
        # was available as of R. This prevents restated prior-quarter
        # financials (published after R) from leaking into the computation.
        # ---------------------------------------------------------------
        def _prev_quarter_end(d: pd.Timestamp) -> pd.Timestamp:
            m, y = d.month, d.year
            if m == 3:  return pd.Timestamp(y - 1, 12, 31)
            if m == 6:  return pd.Timestamp(y, 3, 31)
            if m == 9:  return pd.Timestamp(y, 6, 30)
            return pd.Timestamp(y, 9, 30)

        df["prev_date"] = df["date"].map(_prev_quarter_end)

        # Build lookup table: (stock_code, prev_date, prev_report_date) → NetAsset_begin
        prev_lkp = (
            df[["stock_code", "date", "report_date", "NetAsset_end"]]
            .rename(columns={"date": "prev_date",
                             "report_date": "prev_report_date",
                             "NetAsset_end": "NetAsset_begin"})
            .sort_values(["stock_code", "prev_date", "prev_report_date"])
        )

        # Merge: join each record with all versions of its previous period,
        # then keep only the latest prev version published by current report_date.
        merged = df.merge(prev_lkp, on=["stock_code", "prev_date"], how="left")
        merged = merged[
            merged["prev_report_date"].isna() |
            (merged["prev_report_date"] <= merged["report_date"])
        ]
        merged = (
            merged
            .sort_values(["stock_code", "date", "report_date", "prev_report_date"])
            .groupby(["stock_code", "date", "report_date"], as_index=False)
            .last()
        )
        df = merged.drop(columns=["prev_date", "prev_report_date"], errors="ignore")

        # ---------------------------------------------------------------
        # Step 6: AvgNetAsset and Net_ROE
        # ---------------------------------------------------------------
        df["AvgNetAsset"] = (df["NetAsset_begin"] + df["NetAsset_end"]) / 2.0

        # Drop rows where computation is impossible
        df = df[
            df["q_ps_npatoshopc_c"].notna() &
            df["AvgNetAsset"].notna() &
            (df["AvgNetAsset"] != 0.0)
        ].copy()

        df["Net_ROE"] = df["q_ps_npatoshopc_c"] / df["AvgNetAsset"]

        if df.empty:
            empty_vals = np.empty((0, len(trading_dates)), dtype=np.float64)
            return FactorData(
                values=empty_vals,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype='datetime64[ns]'),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        # ---------------------------------------------------------------
        # Step 7: Forward-fill from report_date to daily panel
        # (report_date <= end_date already guaranteed by Step 4)
        # (same approach as FundamentalData.get_daily_panel)
        # ---------------------------------------------------------------
        pivot = df.pivot_table(
            index="symbol",
            columns="report_date",
            values="Net_ROE",
            aggfunc="last",
        )

        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1)
        panel = panel.reindex(columns=trading_dates)

        values = panel.values.astype(np.float64)
        symbols_arr = np.array(panel.index.tolist())
        dates_arr = np.array(panel.columns.tolist(), dtype='datetime64[ns]')

        nan_ratio = np.isnan(values).sum() / values.size if values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"Net_ROE NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=values,
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python roe.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("Net_ROE factor smoke test")
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

    print(f"\n[Step 2] Compute Net_ROE factor")
    calculator = ROE()
    result = calculator.calculate(fd)

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols      : {result.symbols.tolist()}")
    print(f"Date range   : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # ---- Smoke-test assertions ----------------------------------------
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, \
        f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), \
        "Idempotency failed"

    print(f"\n[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # ---- Sample values ---------------------------------------------------
    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {np.round(last5, 4).tolist()}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid : {len(valid_cs)}")
        print(f"  mean    : {valid_cs.mean():.4f}")
        print(f"  median  : {np.median(valid_cs):.4f}")
        print(f"  min     : {valid_cs.min():.4f}")
        print(f"  max     : {valid_cs.max():.4f}")

    # --- Leakage detection ---
    print(f"\n[Step 3] Leakage detection (5 split ratios)")
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
