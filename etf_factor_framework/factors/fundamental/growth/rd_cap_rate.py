"""
RD_CAP_RATE factor (R&D Capitalization Rate)

Formula: CapRate = CapitalizedRD / (ExpensedRD + CapitalizedRD)
  - CapitalizedRD: quarterly increment in BS development expenditure
                   = q_bs_rade_t[t] - q_bs_rade_t[t-1], clipped to >= 0
  - ExpensedRD:    single-quarter R&D expense from income statement = q_ps_rade_c

Both fields from lixinger.financial_statements.
Result is forward-filled from report_date to trading dates (max 180 days).

Factor direction: negative (high capitalization rate may signal earnings management)
Factor category: growth - R&D quality

Boundary conditions:
  - TotalRD <= 0 or is NaN    -> NaN  (no R&D; undefined ratio)
  - bs_diff < 0 (BS decreased) -> CapitalizedRD = 0  (development costs transferred out)
  - CapRate range [0, 1] by construction (CapitalizedRD clipped to >= 0)
  - Companies without R&D disclosure (NaN q_ps_rade_c) -> NaN
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "RD_CAP_RATE"
FACTOR_DIRECTION = -1  # negative: high capitalization rate -> potential earnings management


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class RD_CAP_RATE(FundamentalFactorCalculator):
    """
    R&D Capitalization Rate factor.

    RD_CAP_RATE = CapitalizedRD / (CapitalizedRD + ExpensedRD)
    where:
      CapitalizedRD = max(0, q_bs_rade_t[t] - q_bs_rade_t[t-1])
                     (quarterly BS development expenditure increment)
      ExpensedRD    = q_ps_rade_c (single-quarter R&D expense from income statement)

    Companies without R&D disclosure (NaN q_ps_rade_c) receive NaN factor values.
    TotalRD = 0 (no R&D at all) also receives NaN.
    Universe restricted to A-share mainboard stocks (60xxxx / 00xxxx).
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
        Compute RD_CAP_RATE daily panel (N stocks x T days).

        Returns:
            FactorData with values = CapitalizedRD / TotalRD (ratio in [0, 1])
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        cap_rate_v, cap_rate_s, _ = self._build_cap_rate_panel(
            fundamental_data, trading_dates_arr
        )

        if cap_rate_v.size == 0:
            raise ValueError("RD_CAP_RATE: cap_rate panel is empty")

        # Apply A-share mainboard filter
        mainboard_mask = np.array([_is_mainboard(s) for s in cap_rate_s])
        cap_rate_v = cap_rate_v[mainboard_mask]
        cap_rate_s = cap_rate_s[mainboard_mask]

        if len(cap_rate_s) == 0:
            raise ValueError("RD_CAP_RATE: no mainboard symbols after filtering")

        cap_rate_v = cap_rate_v.astype(np.float64)

        nan_ratio = np.isnan(cap_rate_v).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"RD_CAP_RATE NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=cap_rate_v,
            symbols=cap_rate_s,
            dates=trading_dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal helper: cap_rate panel
    # ------------------------------------------------------------------

    def _build_cap_rate_panel(
        self,
        fundamental_data: FundamentalData,
        target_dates: np.ndarray,
    ):
        """
        Build daily cap_rate panel by computing per-quarter capitalization rate
        and forward-filling from report_date.

        CapRate = max(0, BS_diff) / (max(0, BS_diff) + expensed_rd)
        where BS_diff = q_bs_rade_t[t] - q_bs_rade_t[t-1] (sorted by financial period date).

        Forward-fills from report_date (max 180 trading days).

        Returns:
            (values np.ndarray (N, T), symbols np.ndarray (N,), dates np.ndarray (T,))
        """
        raw = fundamental_data._raw_data
        trading_dates = pd.DatetimeIndex(target_dates)
        empty = (
            np.empty((0, len(trading_dates)), dtype=np.float64),
            np.array([], dtype=object),
            np.array(trading_dates, dtype="datetime64[ns]"),
        )

        if raw is None or raw.empty:
            return empty

        for field in ["q_ps_rade_c", "q_bs_rade_t"]:
            if field not in raw.columns:
                return empty

        df = raw[["stock_code", "symbol", "date", "report_date",
                  "q_ps_rade_c", "q_bs_rade_t"]].copy()

        df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df[df["report_date"] <= fundamental_data.end_date]

        if df.empty:
            return empty

        # Sort by (stock_code, financial period date) for BS diff
        df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

        # Compute quarterly BS increment: current - previous period balance
        # clip to >= 0 (negative diff means costs transferred to intangible assets)
        # PIT correctness: use the most recent prior period whose report_date is
        # already published (report_date <= current period's report_date). This handles
        # late-filing annual reports that would otherwise cause look-ahead bias.
        # We check shift(1) first, then fall back to shift(2) if shift(1) is late-filed.
        df["bs_prev_1"] = df.groupby("stock_code")["q_bs_rade_t"].shift(1)
        df["rd_prev_1"] = df.groupby("stock_code")["report_date"].shift(1)
        df["bs_prev_2"] = df.groupby("stock_code")["q_bs_rade_t"].shift(2)
        df["rd_prev_2"] = df.groupby("stock_code")["report_date"].shift(2)
        df["bs_prev"] = np.where(
            df["rd_prev_1"].notna() & (df["rd_prev_1"] <= df["report_date"]),
            df["bs_prev_1"],
            np.where(
                df["rd_prev_2"].notna() & (df["rd_prev_2"] <= df["report_date"]),
                df["bs_prev_2"],
                np.nan,
            ),
        )
        df["capitalized_rd"] = (df["q_bs_rade_t"] - df["bs_prev"]).clip(lower=0)

        # expensed_rd = single-quarter R&D expense from income statement
        df["expensed_rd"] = df["q_ps_rade_c"]

        # total_rd and cap_rate
        df["total_rd"] = df["capitalized_rd"] + df["expensed_rd"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["cap_rate"] = np.where(
                df["total_rd"].notna() & (df["total_rd"] > 0),
                df["capitalized_rd"] / df["total_rd"],
                np.nan,
            )

        # Keep only rows with valid cap_rate, use report_date as signal date
        cap_df = df[["symbol", "report_date", "cap_rate"]].dropna(
            subset=["symbol", "cap_rate"]
        ).copy()

        if cap_df.empty:
            return empty

        cap_df["report_date"] = pd.to_datetime(cap_df["report_date"])

        # Pivot: rows=symbol, cols=report_date, values=cap_rate
        pivot = cap_df.pivot_table(
            index="symbol",
            columns="report_date",
            values="cap_rate",
            aggfunc="last",
        )

        # Forward-fill to trading dates (max 180 days ~ 3 quarters)
        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1, limit=180).reindex(columns=trading_dates)

        values = panel.values.astype(np.float64)

        sorted_syms = sorted(panel.index.tolist())
        sorted_idx = np.argsort(panel.index.tolist())
        values = values[sorted_idx]

        symbols_arr = np.array(sorted_syms)
        dates_arr = np.array(panel.columns.tolist(), dtype="datetime64[ns]")

        return values, symbols_arr, dates_arr


# ------------------------------------------------------------------
# Smoke test (run: python rd_cap_rate.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("RD_CAP_RATE factor smoke test")
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

    print(f"\n[Step 2] Data validation: check required fields")
    for field in ["q_ps_rade_c", "q_bs_rade_t"]:
        v, s, d = fd.get_daily_panel(field)
        nr = np.isnan(v).mean()
        print(f"  {field}: shape={v.shape}, NaN={nr:.1%}")
        assert v.size > 0, f"{field} returned empty array"

    print(f"\n[Step 3] Compute RD_CAP_RATE factor")
    calculator = RD_CAP_RATE()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
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

    print(f"\n[Step 4] Smoke test assertions")
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid_vals = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid_vals).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Leakage detection ---
    print(f"\n[Step 5] Leakage detection (5 split ratios)")
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
