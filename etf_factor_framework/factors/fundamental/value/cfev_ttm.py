"""
CFEV_TTM factor (Cash Flow to Enterprise Value, TTM)

Formula: CFEV_TTM = OCF_TTM / EV
  - OCF_TTM = mc / pcf_ttm
      mc      : lixinger.fundamental daily total market cap (yuan)
      pcf_ttm : lixinger.fundamental daily PCF-TTM (market cap / TTM operating cash flow)
      => OCF_TTM = mc / pcf_ttm  (yuan, TTM operating cash flow)
  - EV = mc + TL + MI - Cash
      mc   : total market cap (yuan, daily from fundamental)
      TL   : q_bs_tl_t   (total liabilities, yuan, MRQ, from financial_statements)
      MI   : q_bs_etmsh_t (minority interest, yuan, MRQ, from financial_statements)
      Cash : q_bs_cabb_t  (monetary funds, yuan, MRQ, from financial_statements)
      (Preferred stock = 0 in A-shares)

Factor direction: positive (higher CFEV_TTM = more operating cash flow per unit EV = cheaper)
Factor category: value - denominator-improved value

Note on units: lixinger.fundamental stores mc in yuan (CNY).
financial_statements fields are also in yuan. No unit conversion needed.

Boundary conditions:
  - |pcf_ttm| < 1e-6: OCF_TTM = NaN (near-zero or degenerate)
  - OCF_TTM <= 0 (negative operating cash flow): NaN
  - EV < 1e4 yuan (degenerate): NaN
  - EV <= 0 (highly-leveraged, cash > mc + debt): NaN

Mainboard filter: SHSE.60xxxx and SZSE.00xxxx only.
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

FACTOR_NAME = "CFEV_TTM"
FACTOR_DIRECTION = 1   # positive: higher CFEV_TTM (cheaper on OCF/EV basis) is better
MIN_ABS_PCF = 1e-6     # |pcf_ttm| below this => degenerate => NaN
MIN_EV = 1e4           # EV below this (yuan) => degenerate => NaN


def _is_mainboard(symbol: str) -> bool:
    """Return True if symbol belongs to A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r'(\d{6})', symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith('60') or code.startswith('00')


def _align_to(src_v, src_s, src_d, tgt_s, tgt_d):
    """
    Align panel (src_v: N_src x T_src) to target (tgt_s, tgt_d).
    Symbols/dates must be sorted for searchsorted to work correctly.
    Returns aligned array (N_tgt x T_tgt), filled with NaN where not found.
    """
    out = np.full((len(tgt_s), len(tgt_d)), np.nan, dtype=np.float64)

    common_s = np.intersect1d(src_s, tgt_s)
    if len(common_s) == 0:
        return out

    src_si = np.searchsorted(src_s, common_s)
    tgt_si = np.searchsorted(tgt_s, common_s)

    # Map dates
    di = np.searchsorted(src_d, tgt_d)
    valid_di = di < len(src_d)
    valid_match = valid_di & (src_d[np.minimum(di, len(src_d) - 1)] == tgt_d)

    if not valid_match.any():
        return out

    src_cols = di[valid_match]
    tgt_cols = np.where(valid_match)[0]
    out[np.ix_(tgt_si, tgt_cols)] = src_v[np.ix_(src_si, src_cols)]
    return out


class CFEV_TTM(FundamentalFactorCalculator):
    """
    Cash Flow to Enterprise Value (TTM) factor.

    CFEV_TTM = OCF_TTM / EV
    where:
      OCF_TTM = mc (yuan) / pcf_ttm  (TTM operating cash flow from market-implied PCF)
      EV      = mc + TL + MI - Cash  (enterprise value, yuan)

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
    Stocks with negative/zero OCF_TTM or EV < 1e4 are set to NaN.
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
        Calculate CFEV_TTM daily panel.

        Returns:
            FactorData: N stocks x T days, values = OCF_TTM / EV (float64)
        """
        # --- 1. Load daily valuation panels from lixinger.fundamental ---
        mc_v, mc_s, mc_d = fundamental_data.get_valuation_panel("mc")
        pcf_v, pcf_s, pcf_d = fundamental_data.get_valuation_panel("pcf_ttm")

        if mc_v.size == 0:
            raise ValueError("CFEV_TTM: get_valuation_panel('mc') returned empty array")
        if pcf_v.size == 0:
            raise ValueError("CFEV_TTM: get_valuation_panel('pcf_ttm') returned empty array")

        # --- 2. Load quarterly financial panels (forward-filled to daily) ---
        tl_v, tl_s, tl_d = fundamental_data.get_daily_panel("q_bs_tl_t")
        mi_v, mi_s, mi_d = fundamental_data.get_daily_panel("q_bs_etmsh_t")
        cash_v, cash_s, cash_d = fundamental_data.get_daily_panel("q_bs_cabb_t")

        # --- 3. Determine universe: intersection of mc and pcf_ttm, then mainboard filter ---
        mc_s_sorted = np.sort(mc_s)
        pcf_s_sorted = np.sort(pcf_s)

        base_syms = np.intersect1d(mc_s_sorted, pcf_s_sorted)
        mainboard_mask = np.array([_is_mainboard(s) for s in base_syms])
        common_syms = base_syms[mainboard_mask]

        if len(common_syms) == 0:
            raise ValueError("CFEV_TTM: no mainboard stocks in mc ∩ pcf_ttm universe")

        # common dates = mc_d (trading calendar; all valuation panels share it)
        common_dates = mc_d

        # --- 4. Align mc and pcf_ttm to (common_syms, common_dates) ---
        mc_s_si = np.argsort(mc_s)
        mc_aligned = _align_to(mc_v[mc_s_si], mc_s[mc_s_si], mc_d, common_syms, common_dates)

        pcf_s_si = np.argsort(pcf_s)
        pcf_aligned = _align_to(pcf_v[pcf_s_si], pcf_s[pcf_s_si], pcf_d, common_syms, common_dates)

        # --- 5. Align quarterly panels (already forward-filled to daily in get_daily_panel) ---
        tl_s_si = np.argsort(tl_s)
        tl_aligned = _align_to(tl_v[tl_s_si], tl_s[tl_s_si], tl_d, common_syms, common_dates)

        mi_s_si = np.argsort(mi_s)
        mi_aligned = _align_to(mi_v[mi_s_si], mi_s[mi_s_si], mi_d, common_syms, common_dates)

        cash_s_si = np.argsort(cash_s)
        cash_aligned = _align_to(cash_v[cash_s_si], cash_s[cash_s_si], cash_d, common_syms, common_dates)

        # --- 6. Compute OCF_TTM = mc / pcf_ttm ---
        with np.errstate(divide="ignore", invalid="ignore"):
            ocf_ttm = np.where(
                np.abs(pcf_aligned) >= MIN_ABS_PCF,
                mc_aligned / pcf_aligned,
                np.nan,
            )
        # Only positive OCF_TTM is meaningful; negative cash flow => NaN
        ocf_ttm = np.where(ocf_ttm > 0, ocf_ttm, np.nan)

        # --- 7. Compute EV = mc + TL + MI - Cash ---
        # MI and Cash: treat NaN as 0 (coverage > 99%, impact is secondary)
        mi_filled = np.where(np.isnan(mi_aligned), 0.0, mi_aligned)
        cash_filled = np.where(np.isnan(cash_aligned), 0.0, cash_aligned)
        ev = mc_aligned + tl_aligned + mi_filled - cash_filled

        # --- 8. Compute CFEV_TTM = OCF_TTM / EV ---
        with np.errstate(divide="ignore", invalid="ignore"):
            cfev = np.where(
                (ev >= MIN_EV) & ~np.isnan(ocf_ttm) & ~np.isnan(tl_aligned),
                ocf_ttm / ev,
                np.nan,
            )
        cfev = cfev.astype(np.float64)

        # Replace any remaining inf
        cfev = np.where(np.isinf(cfev), np.nan, cfev)

        nan_ratio = np.isnan(cfev).mean() if cfev.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"CFEV_TTM NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=cfev,
            symbols=common_syms,
            dates=common_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python cfev_ttm.py)
# Uses full market with short date range for cross-section adequacy.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("CFEV_TTM factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute CFEV_TTM factor")
    calculator = CFEV_TTM()
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

    # Sanity: check known stocks
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]
    fd_test = FundamentalData(
        start_date="2024-01-01",
        end_date="2024-12-31",
        stock_codes=TEST_CODES,
    )
    result_test = calculator.calculate(fd_test)
    print(f"\nSample values (5 test stocks, last 5 dates):")
    target_syms = {"SHSE.600519", "SZSE.000858", "SHSE.601318", "SZSE.000333", "SHSE.600036"}
    for i, sym in enumerate(result_test.symbols):
        if sym in target_syms:
            row = result_test.values[i]
            last5 = row[-5:]
            print(f"  {sym}: {np.round(last5, 6)}")

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
