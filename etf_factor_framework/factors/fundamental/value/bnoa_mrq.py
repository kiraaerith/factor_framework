"""
BNOA_MRQ factor (Book value of Net Operating Assets to Market value, MRQ)

Formula: BNOA_MRQ = NOA_book / NOA_market
  - NOA_book   = q_bs_tetoshopc_t - fin_net_assets  (yuan)
      q_bs_tetoshopc_t :归属母公司普通股股东权益合计 (MRQ, yuan, from financial_statements)
  - NOA_market = mc - fin_net_assets  (yuan)
      mc           : lixinger.fundamental daily total market cap (yuan)
  - fin_net_assets = fin_assets - fin_liabilities  (yuan, from financial_statements)
      fin_assets    = q_bs_cabb_t + q_bs_tfa_t + q_bs_cdfa_t + q_bs_cri_t
                    + q_bs_ocri_t + q_bs_oeii_t + q_bs_oncfa_t + q_bs_rei_t
      fin_liabilities = q_bs_stl_t + q_bs_ltl_t + q_bs_bp_t

Units: all financial_statements fields and mc are in yuan (CNY).
No unit conversion needed; ratio is dimensionless.

Factor direction: positive
Factor category: value - denominator-improved value

Boundary conditions:
  - NOA_market <= 0            -> NaN  (non-positive denominator)
  - q_bs_tetoshopc_t NaN       -> NaN
  - mc NaN                     -> NaN
  - fin_net_assets NaN         -> NaN  (all fin asset/liab items NaN)
  - Financial industry (银行/非银金融) -> NaN (NOA concept not applicable)

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

FACTOR_NAME = "BNOA_MRQ"
FACTOR_DIRECTION = 1   # positive: higher BNOA_MRQ is better

FIN_ASSET_FIELDS = [
    "q_bs_cabb_t",    # monetary funds
    "q_bs_tfa_t",     # trading financial assets
    "q_bs_cdfa_t",    # derivative financial assets
    "q_bs_cri_t",     # debt investments
    "q_bs_ocri_t",    # other debt investments
    "q_bs_oeii_t",    # other equity instrument investments
    "q_bs_oncfa_t",   # other non-current financial assets
    "q_bs_rei_t",     # investment properties
]

FIN_LIAB_FIELDS = [
    "q_bs_stl_t",     # short-term borrowings
    "q_bs_ltl_t",     # long-term borrowings
    "q_bs_bp_t",      # bonds payable
]

FINANCIAL_INDUSTRIES = {"银行", "非银金融"}


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

    di = np.searchsorted(src_d, tgt_d)
    valid_di = di < len(src_d)
    valid_match = valid_di & (src_d[np.minimum(di, len(src_d) - 1)] == tgt_d)

    if not valid_match.any():
        return out

    src_cols = di[valid_match]
    tgt_cols = np.where(valid_match)[0]
    out[np.ix_(tgt_si, tgt_cols)] = src_v[np.ix_(src_si, src_cols)]
    return out


class BNOA_MRQ(FundamentalFactorCalculator):
    """
    Book value of Net Operating Assets to Market value (MRQ) factor.

    BNOA_MRQ = NOA_book / NOA_market
    where:
      NOA_book   = q_bs_tetoshopc_t - fin_net_assets  (yuan)
      NOA_market = mc - fin_net_assets                (yuan)
      fin_net_assets = fin_assets - fin_liabilities   (yuan, balance sheet items)

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
    Financial industry stocks (银行/非银金融) are set to NaN.
    NOA_market <= 0 is set to NaN.
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
        Calculate BNOA_MRQ daily panel.

        Returns:
            FactorData: N stocks x T days, values = NOA_book / NOA_market (float64)
        """
        # --- 1. Load equity panel (q_bs_tetoshopc_t, quarterly ffill to daily) ---
        eq_v, eq_s, eq_d = fundamental_data.get_daily_panel("q_bs_tetoshopc_t")
        if eq_v.size == 0:
            raise ValueError("BNOA_MRQ: get_daily_panel('q_bs_tetoshopc_t') returned empty array")

        # --- 2. Load daily market cap ---
        mc_v, mc_s, mc_d = fundamental_data.get_valuation_panel("mc")
        if mc_v.size == 0:
            raise ValueError("BNOA_MRQ: get_valuation_panel('mc') returned empty array")

        # --- 3. Load balance sheet fields (forward-filled quarterly to daily) ---
        fa_panels = {}
        for f in FIN_ASSET_FIELDS:
            fa_panels[f] = fundamental_data.get_daily_panel(f)

        fl_panels = {}
        for f in FIN_LIAB_FIELDS:
            fl_panels[f] = fundamental_data.get_daily_panel(f)

        # --- 4. Determine common universe: eq ∩ mc, mainboard filter ---
        base_syms = np.intersect1d(np.sort(eq_s), np.sort(mc_s))
        mainboard_mask = np.array([_is_mainboard(s) for s in base_syms])
        common_syms = base_syms[mainboard_mask]

        if len(common_syms) == 0:
            raise ValueError("BNOA_MRQ: no mainboard stocks in eq ∩ mc universe")

        common_dates = mc_d  # trading calendar from mc (daily)

        # --- 5. Align equity panel to (common_syms, common_dates) ---
        eq_s_si = np.argsort(eq_s)
        eq_aligned = _align_to(eq_v[eq_s_si], eq_s[eq_s_si], eq_d, common_syms, common_dates)

        # --- 6. Align mc to (common_syms, common_dates) ---
        mc_s_si = np.argsort(mc_s)
        mc_aligned = _align_to(mc_v[mc_s_si], mc_s[mc_s_si], mc_d, common_syms, common_dates)

        # --- 7. Align and sum balance sheet items ---
        fa_aligned_list = []
        for f in FIN_ASSET_FIELDS:
            v, s, d = fa_panels[f]
            if v.size > 0:
                s_si = np.argsort(s)
                fa_aligned_list.append(_align_to(v[s_si], s[s_si], d, common_syms, common_dates))
            else:
                fa_aligned_list.append(np.full((len(common_syms), len(common_dates)), np.nan))

        fl_aligned_list = []
        for f in FIN_LIAB_FIELDS:
            v, s, d = fl_panels[f]
            if v.size > 0:
                s_si = np.argsort(s)
                fl_aligned_list.append(_align_to(v[s_si], s[s_si], d, common_syms, common_dates))
            else:
                fl_aligned_list.append(np.full((len(common_syms), len(common_dates)), np.nan))

        # Stack and sum (treat NaN as 0 per item; flag all-NaN groups as NaN)
        fa_stack = np.stack(fa_aligned_list, axis=0)   # (8, N, T)
        fl_stack = np.stack(fl_aligned_list, axis=0)   # (3, N, T)

        all_nan_assets = np.all(np.isnan(fa_stack), axis=0)   # (N, T)
        all_nan_liabs  = np.all(np.isnan(fl_stack), axis=0)   # (N, T)

        fin_assets = np.nansum(fa_stack, axis=0)   # (N, T), 0 where all-NaN
        fin_assets = np.where(all_nan_assets, np.nan, fin_assets)

        fin_liabs = np.nansum(fl_stack, axis=0)    # (N, T), 0 where all-NaN
        fin_liabs = np.where(all_nan_liabs, np.nan, fin_liabs)

        # fin_net_assets = fin_assets - fin_liabilities (NaN if either side is NaN)
        fin_net_assets = fin_assets - fin_liabs

        # --- 8. Compute NOA_book and NOA_market ---
        noa_book   = eq_aligned - fin_net_assets     # (N, T), yuan
        noa_market = mc_aligned - fin_net_assets     # (N, T), yuan

        # --- 9. Compute BNOA_MRQ ---
        with np.errstate(divide="ignore", invalid="ignore"):
            bnoa = np.where(
                (noa_market > 0)
                & ~np.isnan(noa_book)
                & ~np.isnan(mc_aligned),
                noa_book / noa_market,
                np.nan,
            )
        bnoa = bnoa.astype(np.float64)

        # Replace any remaining inf
        bnoa = np.where(np.isinf(bnoa), np.nan, bnoa)

        # --- 10. Financial industry: force NaN ---
        industry_map = fundamental_data.get_industry_map()
        for i, sym in enumerate(common_syms):
            if industry_map.get(sym) in FINANCIAL_INDUSTRIES:
                bnoa[i, :] = np.nan

        nan_ratio = np.isnan(bnoa).mean() if bnoa.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"BNOA_MRQ NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=bnoa,
            symbols=common_syms,
            dates=common_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python bnoa_mrq.py)
# Uses full market with short date range for cross-section adequacy.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("BNOA_MRQ factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute BNOA_MRQ factor")
    calculator = BNOA_MRQ()
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
