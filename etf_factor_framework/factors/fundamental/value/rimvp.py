"""
RIMVP factor (Residual Income Model Value to Price)

RIMVP = RIMV / mc

where RIMV is computed via Edwards-Bell-Ohlson (EBO) model:
  RIMV = B0 + sum_{t=1}^{3} [(ROE_FYt - r) / (1+r)^t * B_{t-1}]

  B0   = mc / pb                          (book equity, in yuan)
  ROE_FYt = net_profit_FYt / B_{t-1}     (forward ROE from analyst forecasts)
  B_t  = B_{t-1} * (1 + ROE_FYt * (1 - DPR))
  DPR  = clip(dyr * pe_ttm, 0, 1)        (dividend payout ratio)
  r    = 0.12                             (fixed discount rate)

Data sources:
  - lixinger.fundamental : mc, pb, pe_ttm, dyr  (daily)
  - tushare.report_rc    : np (万元), quarter (YYYYQ4 annual forecasts)

Unit note:
  - mc is stored in yuan (元) in lixinger.fundamental
  - np is stored in 万元 (10k yuan) in tushare.report_rc
  - All intermediate values in yuan; RIMVP is dimensionless

Factor direction: positive (higher RIMVP = more undervalued)
Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only

Fallback: stocks with no analyst coverage use RIMV = B0 (degrades to BP_MRQ).
"""

import os
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # etf_cross_ml-master
TUSHARE_DB = str(_PROJECT_ROOT.parent / "china_stock_data" / "tushare.db")

FACTOR_NAME = "RIMVP"
FACTOR_DIRECTION = 1   # positive: higher RIMVP (more undervalued) is better
DISCOUNT_RATE = 0.12   # fixed annual discount rate


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _align_panel(
    base_symbols: np.ndarray,
    other_symbols: np.ndarray,
    other_values: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    """Reindex other_values (N_other x T) to base_symbols -> (N_base x T)."""
    if np.array_equal(base_symbols, other_symbols):
        return other_values
    other_dict = {s: i for i, s in enumerate(other_symbols)}
    result = np.full((len(base_symbols), n_dates), np.nan, dtype=np.float64)
    for j, sym in enumerate(base_symbols):
        if sym in other_dict:
            result[j] = other_values[other_dict[sym]]
    return result


class RIMVP(FundamentalFactorCalculator):
    """
    Residual Income Model Value to Price factor.

    Stocks with analyst coverage (tushare.report_rc) use the full EBO model.
    Stocks without coverage fall back to RIMV = B0, i.e. RIMVP = 1/pb (= BP_MRQ).
    Stocks with negative book value (pb <= 0) are set to NaN.
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"direction": FACTOR_DIRECTION, "r": DISCOUNT_RATE}

    def calculate(self, fd: FundamentalData) -> FactorData:
        """Compute RIMVP daily panel."""
        # ------------------------------------------------------------------
        # Step 1: Load lixinger daily panels
        # ------------------------------------------------------------------
        mc_v, mc_s, dates = fd.get_valuation_panel("mc")
        pb_v, pb_s, _ = fd.get_valuation_panel("pb")
        pe_v, pe_s, _ = fd.get_valuation_panel("pe_ttm")
        dyr_v, dyr_s, _ = fd.get_valuation_panel("dyr")

        if mc_v.size == 0:
            raise ValueError("RIMVP: get_valuation_panel('mc') returned empty array")

        # ------------------------------------------------------------------
        # Step 2: Align all panels to mc_s as base symbol universe
        # ------------------------------------------------------------------
        n_dates = len(dates)
        pb_a = _align_panel(mc_s, pb_s, pb_v, n_dates)
        pe_a = _align_panel(mc_s, pe_s, pe_v, n_dates)
        dyr_a = _align_panel(mc_s, dyr_s, dyr_v, n_dates)

        # ------------------------------------------------------------------
        # Step 3: Mainboard filter
        # ------------------------------------------------------------------
        mb_mask = np.array([_is_mainboard(s) for s in mc_s])
        mc_v = mc_v[mb_mask]
        pb_a = pb_a[mb_mask]
        pe_a = pe_a[mb_mask]
        dyr_a = dyr_a[mb_mask]
        symbols = mc_s[mb_mask]

        # ------------------------------------------------------------------
        # Step 4: Compute B0 (yuan) and DPR
        # ------------------------------------------------------------------
        # B0 = mc / pb  [yuan]; NaN when pb <= 0 (negative book value) or pb/mc is NaN
        with np.errstate(invalid='ignore', divide='ignore'):
            B0 = np.where(
                np.isnan(pb_a) | (pb_a <= 0) | np.isnan(mc_v),
                np.nan,
                mc_v / pb_a,
            )

        # DPR = clip(dyr * pe_ttm, 0, 1)
        # Set DPR = 0 when pe_ttm <= 0 (loss-making) or any operand is NaN
        with np.errstate(invalid='ignore'):
            dpr_raw = dyr_a * pe_a
        DPR = np.where(
            np.isnan(dpr_raw) | np.isnan(pe_a) | (pe_a <= 0),
            0.0,
            np.clip(dpr_raw, 0.0, 1.0),
        )

        # ------------------------------------------------------------------
        # Step 5: Load analyst consensus forecasts (tushare.report_rc)
        # Returns np1, np2, np3: shape (N, T), units yuan (元)
        # NaN where no analyst coverage
        # ------------------------------------------------------------------
        np1, np2, np3 = self._load_analyst_panel(fd, symbols, dates)

        # ------------------------------------------------------------------
        # Step 6: EBO model - compute RIMV
        # ------------------------------------------------------------------
        r = DISCOUNT_RATE

        # ROE_FY1 = np1 / B0; NaN when B0 <= 0 or np1 is NaN
        with np.errstate(invalid='ignore', divide='ignore'):
            ROE1 = np.where(
                np.isnan(B0) | (B0 <= 0) | np.isnan(np1),
                np.nan,
                np1 / B0,
            )

        B1 = B0 * (1.0 + ROE1 * (1.0 - DPR))

        with np.errstate(invalid='ignore', divide='ignore'):
            ROE2 = np.where(
                np.isnan(B1) | (B1 <= 0) | np.isnan(np2),
                np.nan,
                np2 / B1,
            )

        B2 = B1 * (1.0 + ROE2 * (1.0 - DPR))

        with np.errstate(invalid='ignore', divide='ignore'):
            ROE3 = np.where(
                np.isnan(B2) | (B2 <= 0) | np.isnan(np3),
                np.nan,
                np3 / B2,
            )

        excess1 = (ROE1 - r) / (1.0 + r) ** 1 * B0
        excess2 = (ROE2 - r) / (1.0 + r) ** 2 * B1
        excess3 = (ROE3 - r) / (1.0 + r) ** 3 * B2

        RIMV_full = B0 + excess1 + excess2 + excess3

        # Fallback: RIMV = B0 wherever the full EBO calculation yields NaN
        # (covers both "no analyst coverage" and numerical failures in EBO)
        RIMV = np.where(np.isnan(RIMV_full), B0, RIMV_full)

        # ------------------------------------------------------------------
        # Step 7: RIMVP = RIMV / mc
        # ------------------------------------------------------------------
        with np.errstate(invalid='ignore', divide='ignore'):
            rimvp = np.where(
                np.isnan(mc_v) | (mc_v <= 0) | np.isnan(RIMV),
                np.nan,
                RIMV / mc_v,
            )

        rimvp = np.where(np.isinf(rimvp), np.nan, rimvp).astype(np.float64)

        nan_ratio = np.isnan(rimvp).mean() if rimvp.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(f"RIMVP NaN ratio is high: {nan_ratio:.1%}")

        return FactorData(
            values=rimvp,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal: load tushare analyst consensus panel
    # ------------------------------------------------------------------

    def _load_analyst_panel(
        self,
        fd: FundamentalData,
        symbols: np.ndarray,
        dates: np.ndarray,
    ):
        """
        Load 180-day rolling consensus net profit for FY1/FY2/FY3.

        FY1/FY2/FY3 definition (year-end quarters only):
          - month >= 5: FY1 = current year Q4, FY2 = +1, FY3 = +2
          - month <  5: FY1 = (year-1) Q4,    FY2 = +1, FY3 = +2

        Returns:
            (np1, np2, np3): each ndarray shape (N, T), units yuan (元).
            NaN where no analyst coverage.
        """
        N = len(symbols)
        T = len(dates)
        np1 = np.full((N, T), np.nan, dtype=np.float64)
        np2 = np.full((N, T), np.nan, dtype=np.float64)
        np3 = np.full((N, T), np.nan, dtype=np.float64)

        if T == 0 or N == 0:
            return np1, np2, np3

        # Date range for tushare query
        dates_pd = pd.DatetimeIndex(dates)
        query_start = (dates_pd[0] - pd.Timedelta(days=181)).strftime('%Y%m%d')
        query_end = fd.end_date.strftime('%Y%m%d')

        try:
            conn = sqlite3.connect(TUSHARE_DB)
            fc_df = pd.read_sql_query(
                f"""
                SELECT ts_code, report_date, quarter, np
                FROM report_rc
                WHERE report_date BETWEEN '{query_start}' AND '{query_end}'
                  AND np IS NOT NULL
                  AND quarter LIKE '%Q4'
                """,
                conn,
            )
            conn.close()
        except Exception as exc:
            warnings.warn(f"RIMVP: failed to load tushare data: {exc}")
            return np1, np2, np3

        if fc_df.empty:
            return np1, np2, np3

        # Filter to mainboard (tushare code format: '600519.SH', '000858.SZ')
        fc_df['code6'] = fc_df['ts_code'].str[:6]
        mb_mask = fc_df['code6'].apply(
            lambda c: c.startswith('60') or c.startswith('00')
        )
        fc_df = fc_df[mb_mask].copy()
        if fc_df.empty:
            return np1, np2, np3

        # Convert np from 万元 to yuan (元): multiply by 1e4
        fc_df['np_yuan'] = fc_df['np'].astype(float) * 1e4

        # Convert report_date (YYYYMMDD text) to datetime
        fc_df['report_date_dt'] = pd.to_datetime(fc_df['report_date'], format='%Y%m%d')

        # Extract forecast year from quarter string (e.g. '2025Q4' -> 2025)
        fc_df['q_year'] = fc_df['quarter'].str[:4].astype(int)

        # Build code6 -> symbol-index mapping
        code6_to_idx = {}
        for i, sym in enumerate(symbols):
            m = re.search(r'(\d{6})', sym)
            if m:
                code6_to_idx[m.group(1)] = i

        # Sort by report_date for binary search
        fc_df = fc_df.sort_values('report_date_dt').reset_index(drop=True)
        fc_dates_arr = fc_df['report_date_dt'].values.astype('datetime64[D]')
        dates_d = dates.astype('datetime64[D]')

        for t_idx, t in enumerate(dates_d):
            t_pd = pd.Timestamp(t)
            t_180 = t - np.timedelta64(180, 'D')

            # Binary search for 180-day window boundaries
            s_i = int(np.searchsorted(fc_dates_arr, t_180, side='left'))
            e_i = int(np.searchsorted(
                fc_dates_arr, t + np.timedelta64(1, 'D'), side='left'
            ))

            if s_i >= e_i:
                continue

            window = fc_df.iloc[s_i:e_i]

            # Determine FY1/FY2/FY3 year-end quarters for this date
            if t_pd.month >= 5:
                fy_years = [t_pd.year, t_pd.year + 1, t_pd.year + 2]
            else:
                fy_years = [t_pd.year - 1, t_pd.year, t_pd.year + 1]

            # Compute mean np_yuan per (code6, q_year) across all analysts
            consensus = (
                window.groupby(['code6', 'q_year'])['np_yuan']
                .mean()
                .reset_index()
            )

            # Assign FY1/FY2/FY3 values into result panels
            for fy_i, fy_year in enumerate(fy_years):
                fy_sub = consensus[consensus['q_year'] == fy_year][
                    ['code6', 'np_yuan']
                ].copy()
                if fy_sub.empty:
                    continue
                fy_sub['s_idx'] = fy_sub['code6'].map(code6_to_idx)
                fy_sub = fy_sub.dropna(subset=['s_idx'])
                if fy_sub.empty:
                    continue
                s_idx = fy_sub['s_idx'].astype(int).values
                np_vals = fy_sub['np_yuan'].values
                if fy_i == 0:
                    np1[s_idx, t_idx] = np_vals
                elif fy_i == 1:
                    np2[s_idx, t_idx] = np_vals
                else:
                    np3[s_idx, t_idx] = np_vals

        return np1, np2, np3


# ------------------------------------------------------------------
# Smoke test (run: python rimvp.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time

    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("RIMVP factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    t0 = time.time()
    fd = FundamentalData(start_date=TEST_START, end_date=TEST_END)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    print(f"\n[Step 2] Compute RIMVP factor")
    calculator = RIMVP()
    t1 = time.time()
    result = calculator.calculate(fd)
    elapsed = time.time() - t1
    print(f"  Factor computed in {elapsed:.1f}s")

    print(f"\nFactor shape : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range   : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")

    # ---- Smoke-test assertions -------------------------------------------
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
        print(f"  median     : {np.median(valid_cs):.6f}")
        print(f"  min        : {valid_cs.min():.4f}")
        print(f"  max        : {valid_cs.max():.4f}")

    # Sample: 5 known test stocks
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
