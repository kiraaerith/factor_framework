"""
CCP_MRQ factor (Contributed Capital-to-Price, Most Recent Quarter)

Formula:
  contributed_capital_mrq = q_bs_tetoshopc_t - q_bs_rtp_t - q_bs_surr_t
  CCP_MRQ = contributed_capital_mrq / mc

  where:
    q_bs_tetoshopc_t : equity attributable to parent common shareholders (yuan, MRQ)
    q_bs_rtp_t       : retained earnings / undistributed profit (yuan, MRQ, filled 0 if NaN)
    q_bs_surr_t      : surplus reserve (yuan, MRQ, filled 0 if NaN)
    mc               : total market cap (yuan, daily) from lixinger.fundamental

Data fields:
  - q_bs_tetoshopc_t : lixinger.financial_statements (quarterly, forward-filled to daily)
  - q_bs_rtp_t       : lixinger.financial_statements (quarterly, forward-filled to daily)
  - q_bs_surr_t      : lixinger.financial_statements (quarterly, forward-filled to daily)
  - mc               : lixinger.fundamental (daily)

Factor direction: positive (higher CCP = stockholder-injected capital underpriced = better)
Factor category: value - improved value (numerator decomposition)

Notes:
  - contributed_capital approximates: share capital + capital reserve - treasury stock
  - If q_bs_rtp_t or q_bs_surr_t is NaN, substitute 0 (conservative; avoids losing the record)
  - If q_bs_tetoshopc_t is NaN, CCP_MRQ = NaN
  - mc = 0 or NaN => CCP_MRQ = NaN
  - Both mc and contributed_capital are in yuan; ratio is dimensionless
  - Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only
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

FACTOR_NAME = "CCP_MRQ"
FACTOR_DIRECTION = 1   # positive: higher CCP (contributed capital undervalued) is better
MIN_MC = 1e-6          # mc below this threshold => treat as NaN


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


class CCP_MRQ(FundamentalFactorCalculator):
    """
    Contributed Capital-to-Price (MRQ) factor.

    Decomposes parent equity into contributed capital (share capital + capital reserve)
    and retained earnings. CCP_MRQ measures the market's pricing of the shareholder-
    injected capital portion relative to total market cap.

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
    mc = 0 or NaN => CCP_MRQ = NaN.
    q_bs_tetoshopc_t = NaN => CCP_MRQ = NaN (missing equity data).
    q_bs_rtp_t or q_bs_surr_t = NaN => substituted with 0.
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
        Calculate CCP_MRQ daily panel.

        Returns:
            FactorData: N stocks x T days, values = contributed_capital / mc (float64)
        """
        # ------------------------------------------------------------------
        # Step 1: Load panels for three financial statement fields (quarterly,
        #         forward-filled to daily using report_date)
        # ------------------------------------------------------------------
        toe_vals, toe_syms, toe_dates = fundamental_data.get_daily_panel("q_bs_tetoshopc_t")
        rtp_vals, rtp_syms, _         = fundamental_data.get_daily_panel("q_bs_rtp_t")
        surr_vals, surr_syms, _       = fundamental_data.get_daily_panel("q_bs_surr_t")

        # ------------------------------------------------------------------
        # Step 2: Load daily market cap panel (mc, in yuan)
        # ------------------------------------------------------------------
        mc_vals, mc_syms, mc_dates = fundamental_data.get_market_cap_panel()

        # ------------------------------------------------------------------
        # Step 3: Apply mainboard filter to each panel
        # ------------------------------------------------------------------
        def _apply_mb_filter(vals, syms):
            mask = np.array([_is_mainboard(s) for s in syms])
            return vals[mask], np.array(syms)[mask]

        toe_vals, toe_syms   = _apply_mb_filter(toe_vals, toe_syms)
        rtp_vals, rtp_syms   = _apply_mb_filter(rtp_vals, rtp_syms)
        surr_vals, surr_syms = _apply_mb_filter(surr_vals, surr_syms)
        mc_vals, mc_syms     = _apply_mb_filter(mc_vals, mc_syms)

        # ------------------------------------------------------------------
        # Step 4: Build DataFrames and align to a unified symbol × date index
        # ------------------------------------------------------------------
        trading_dates = pd.DatetimeIndex(toe_dates)
        mc_dates_idx  = pd.DatetimeIndex(mc_dates)

        df_toe  = pd.DataFrame(toe_vals,  index=toe_syms,  columns=trading_dates)
        df_rtp  = pd.DataFrame(rtp_vals,  index=rtp_syms,  columns=trading_dates)
        df_surr = pd.DataFrame(surr_vals, index=surr_syms, columns=trading_dates)
        df_mc   = pd.DataFrame(mc_vals,   index=mc_syms,   columns=mc_dates_idx)

        # Union all symbols; financial panels and mc may overlap differently
        all_syms = sorted(
            set(toe_syms.tolist()) | set(rtp_syms.tolist()) |
            set(surr_syms.tolist()) | set(mc_syms.tolist())
        )

        df_toe  = df_toe.reindex(index=all_syms,  columns=trading_dates)
        df_rtp  = df_rtp.reindex(index=all_syms,  columns=trading_dates)
        df_surr = df_surr.reindex(index=all_syms, columns=trading_dates)
        df_mc   = df_mc.reindex(index=all_syms,   columns=trading_dates)

        # ------------------------------------------------------------------
        # Step 5: Compute contributed_capital = tetoshopc - rtp(0 if NaN) - surr(0 if NaN)
        # ------------------------------------------------------------------
        arr_toe  = df_toe.values.astype(np.float64)
        arr_rtp  = np.where(np.isnan(df_rtp.values),  0.0, df_rtp.values.astype(np.float64))
        arr_surr = np.where(np.isnan(df_surr.values), 0.0, df_surr.values.astype(np.float64))
        arr_mc   = df_mc.values.astype(np.float64)

        # If arr_toe is NaN => contributed_capital is NaN (propagated automatically)
        contributed_capital = arr_toe - arr_rtp - arr_surr  # (N, T), yuan

        # ------------------------------------------------------------------
        # Step 6: CCP_MRQ = contributed_capital / mc (both in yuan)
        # ------------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ccp_values = np.where(
                (arr_mc > MIN_MC) & ~np.isnan(arr_mc) & ~np.isnan(contributed_capital),
                contributed_capital / arr_mc,
                np.nan,
            )

        # Remove any residual inf values
        ccp_values = np.where(np.isinf(ccp_values), np.nan, ccp_values)
        ccp_values = ccp_values.astype(np.float64)

        nan_ratio = np.isnan(ccp_values).mean() if ccp_values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"CCP_MRQ NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        symbols_arr = np.array(all_syms)
        dates_arr   = np.array(trading_dates, dtype='datetime64[ns]')

        return FactorData(
            values=ccp_values,
            symbols=symbols_arr,
            dates=dates_arr,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python ccp_mrq.py)
# Uses full market with short date range to ensure enough cross-section.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("CCP_MRQ factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute CCP_MRQ factor")
    calculator = CCP_MRQ()
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

    # Sanity: check known stocks have non-NaN CCP_MRQ
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]
    fd_test = FundamentalData(
        start_date="2024-01-01",
        end_date="2024-12-31",
        stock_codes=TEST_CODES,
    )
    result_test = calculator.calculate(fd_test)
    print(f"\nSample values (5 test stocks, last 5 dates):")
    from factors.fundamental.fundamental_data import lixinger_code_to_symbol
    for code in TEST_CODES:
        sym = lixinger_code_to_symbol(code)
        sym_list = result_test.symbols.tolist()
        if sym in sym_list:
            idx = sym_list.index(sym)
            last5 = result_test.values[idx, -5:]
            print(f"  {sym}: {np.round(last5, 4)}")
        else:
            print(f"  {sym}: not found in result")

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
