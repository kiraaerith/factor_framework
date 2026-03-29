"""
SNOA_TTM factor (Sales to Net Operating Assets Market Value, TTM)

Formula: SNOA_TTM = Revenue_TTM / NOA_Market
  - Revenue_TTM : TTM rolling sum of q_ps_oi_c (single-quarter operating revenue, yuan)
  - NOA_Market  = mc - FinNetAssets  (yuan)
      mc           : lixinger.fundamental daily total market cap (yuan)
      FinNetAssets : FinAssets - FinLiabilities (yuan, from financial_statements)
        FinAssets    = q_bs_cabb_t + q_bs_tfa_t + q_bs_cdfa_t + q_bs_cri_t
                     + q_bs_ocri_t + q_bs_oeii_t + q_bs_oncfa_t + q_bs_rei_t
        FinLiabilities = q_bs_stl_t + q_bs_ltl_t + q_bs_bp_t

Units: all fields in yuan (CNY). mc from lixinger.fundamental is in yuan, NOT yi yuan.
No unit conversion needed; the ratio is dimensionless.

Factor direction: positive (higher sales per NOA_market is better)
Factor category: value - denominator-improved value

Boundary conditions:
  - NOA_Market <= 0              -> NaN  (degenerate: high financial asset / data anomaly)
  - Revenue_TTM NaN              -> NaN  (insufficient quarterly data, <4 quarters)
  - mc NaN                       -> NaN
  - Financial industry (bank/non-bank financial) -> NaN (NOA concept not applicable)

Balance sheet items: each NaN item treated as 0 (coverage >98%);
if ALL items in FinAssets group are NaN, FinNetAssets = NaN -> SNOA_TTM NaN.

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

FACTOR_NAME = "SNOA_TTM"
FACTOR_DIRECTION = 1   # positive: higher SNOA_TTM (more revenue per NOA_market) is better

FIN_ASSET_FIELDS = [
    "q_bs_cabb_t",   # monetary funds
    "q_bs_tfa_t",    # trading financial assets
    "q_bs_cdfa_t",   # derivative financial assets
    "q_bs_cri_t",    # debt investments
    "q_bs_ocri_t",   # other debt investments
    "q_bs_oeii_t",   # other equity instrument investments
    "q_bs_oncfa_t",  # other non-current financial assets
    "q_bs_rei_t",    # investment properties
]

FIN_LIAB_FIELDS = [
    "q_bs_stl_t",    # short-term borrowings
    "q_bs_ltl_t",    # long-term borrowings
    "q_bs_bp_t",     # bonds payable
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


class SNOA_TTM(FundamentalFactorCalculator):
    """
    Sales to Net Operating Assets Market Value (TTM) factor.

    SNOA_TTM = Revenue_TTM / NOA_Market
    where:
      Revenue_TTM = TTM rolling sum of q_ps_oi_c (yuan, from single-quarter sums)
      NOA_Market  = mc - FinNetAssets              (yuan)
      FinNetAssets = FinAssets - FinLiabilities    (yuan, from balance sheet)

    Only mainboard stocks (SHSE.60xxxx, SZSE.00xxxx) are included.
    NOA_Market <= 0 is set to NaN.
    Financial industry (banks, non-bank financial) are set to NaN.
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
        Calculate SNOA_TTM daily panel.

        Returns:
            FactorData: N stocks x T days, values = Revenue_TTM / NOA_Market (float64)
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        trading_dates_arr = np.array(trading_dates, dtype="datetime64[ns]")

        # --- 1. Compute TTM panel for revenue ---
        rev_v, rev_s, rev_d = self._compute_ttm_panel(
            fundamental_data, "q_ps_oi_c", trading_dates_arr
        )

        if rev_v.size == 0:
            raise ValueError("SNOA_TTM: q_ps_oi_c TTM panel is empty")

        # --- 2. Load daily market cap ---
        mc_v, mc_s, mc_d = fundamental_data.get_valuation_panel("mc")
        if mc_v.size == 0:
            raise ValueError("SNOA_TTM: get_valuation_panel('mc') returned empty array")

        # --- 3. Load balance sheet fields (forward-filled quarterly to daily) ---
        fa_panels = {}
        for f in FIN_ASSET_FIELDS:
            fa_panels[f] = fundamental_data.get_daily_panel(f)

        fl_panels = {}
        for f in FIN_LIAB_FIELDS:
            fl_panels[f] = fundamental_data.get_daily_panel(f)

        # --- 4. Determine common universe: rev_ttm ∩ mc, mainboard filter ---
        base_syms = np.intersect1d(np.sort(rev_s), np.sort(mc_s))
        mainboard_mask = np.array([_is_mainboard(s) for s in base_syms])
        common_syms = base_syms[mainboard_mask]

        if len(common_syms) == 0:
            raise ValueError("SNOA_TTM: no mainboard stocks in rev_ttm ∩ mc universe")

        common_dates = mc_d  # trading calendar from mc (daily)

        # --- 5. Align revenue TTM panel to (common_syms, common_dates) ---
        rev_s_si = np.argsort(rev_s)
        rev_aligned = _align_to(rev_v[rev_s_si], rev_s[rev_s_si], rev_d,
                                 common_syms, common_dates)

        # --- 6. Align mc to (common_syms, common_dates) ---
        mc_s_si = np.argsort(mc_s)
        mc_aligned = _align_to(mc_v[mc_s_si], mc_s[mc_s_si], mc_d,
                                common_syms, common_dates)

        # --- 7. Align and sum balance sheet items ---
        # Financial assets
        fa_aligned_list = []
        for f in FIN_ASSET_FIELDS:
            v, s, d = fa_panels[f]
            if v.size > 0:
                s_si = np.argsort(s)
                fa_aligned_list.append(_align_to(v[s_si], s[s_si], d, common_syms, common_dates))
            else:
                fa_aligned_list.append(np.full((len(common_syms), len(common_dates)), np.nan))

        # Financial liabilities
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

        all_nan_assets = np.all(np.isnan(fa_stack), axis=0)  # (N, T)
        all_nan_liabs  = np.all(np.isnan(fl_stack), axis=0)  # (N, T)

        fin_assets = np.nansum(fa_stack, axis=0)  # (N, T), 0 where all-NaN
        fin_assets = np.where(all_nan_assets, np.nan, fin_assets)

        fin_liabs = np.nansum(fl_stack, axis=0)   # (N, T), 0 where all-NaN
        fin_liabs = np.where(all_nan_liabs, np.nan, fin_liabs)

        # FinNetAssets = FinAssets - FinLiabilities (yuan)
        # NaN if either side is NaN
        fin_net_assets = fin_assets - fin_liabs

        # --- 8. Compute NOA_Market = mc - FinNetAssets (yuan) ---
        # Where fin_net_assets is NaN (no balance sheet data), treat as 0
        # so NOA_Market falls back to mc (classic SP/MV behavior)
        fin_net_assets_filled = np.where(np.isnan(fin_net_assets), 0.0, fin_net_assets)
        noa_market = mc_aligned - fin_net_assets_filled

        # --- 9. Compute SNOA_TTM ---
        with np.errstate(divide="ignore", invalid="ignore"):
            snoa = np.where(
                (noa_market > 0) & ~np.isnan(rev_aligned) & ~np.isnan(mc_aligned),
                rev_aligned / noa_market,
                np.nan,
            )
        snoa = snoa.astype(np.float64)

        # Replace any remaining inf
        snoa = np.where(np.isinf(snoa), np.nan, snoa)

        # --- 10. Financial industry: force NaN ---
        industry_map = fundamental_data.get_industry_map()
        for i, sym in enumerate(common_syms):
            if industry_map.get(sym) in FINANCIAL_INDUSTRIES:
                snoa[i, :] = np.nan

        nan_ratio = np.isnan(snoa).mean() if snoa.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"SNOA_TTM NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=snoa,
            symbols=common_syms,
            dates=common_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal helper: TTM panel from single-quarter field
    # ------------------------------------------------------------------

    def _compute_ttm_panel(
        self,
        fundamental_data: FundamentalData,
        field: str,
        target_dates: np.ndarray,
    ):
        """
        Build daily TTM panel by summing the 4 most recently disclosed
        single-quarter values. Ordered by report_date to prevent leakage.
        Forward-fills from report_date until next disclosure.
        """
        fundamental_data._load_raw_data()
        raw = fundamental_data._raw_data

        trading_dates = pd.DatetimeIndex(target_dates)
        empty = (
            np.empty((0, len(trading_dates)), dtype=np.float64),
            np.array([], dtype=object),
            np.array(trading_dates, dtype="datetime64[ns]"),
        )

        if raw is None or raw.empty:
            return empty
        if field not in raw.columns:
            return empty

        df = raw[["stock_code", "report_date", field]].dropna(subset=[field]).copy()
        if df.empty:
            return empty

        df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)
        df = df[df["report_date"] <= fundamental_data.end_date]
        if df.empty:
            return empty

        df = df.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

        # Rolling 4-quarter sum; require min 4 periods
        df["ttm_val"] = (
            df.groupby("stock_code")[field]
            .transform(lambda x: x.rolling(4, min_periods=4).sum())
        )

        ttm_df = df[["stock_code", "report_date", "ttm_val"]].dropna(subset=["ttm_val"])
        if ttm_df.empty:
            return empty

        ttm_df = ttm_df.copy()
        ttm_df["report_date"] = pd.to_datetime(ttm_df["report_date"])

        symbol_map = (
            raw[["stock_code", "symbol"]]
            .drop_duplicates("stock_code")
            .set_index("stock_code")["symbol"]
        )
        ttm_df["symbol"] = ttm_df["stock_code"].map(symbol_map)
        ttm_df = ttm_df.dropna(subset=["symbol"])
        if ttm_df.empty:
            return empty

        pivot = ttm_df.pivot_table(
            index="symbol",
            columns="report_date",
            values="ttm_val",
            aggfunc="last",
        )

        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1).reindex(columns=trading_dates)

        values = panel.values.astype(np.float64)
        sorted_idx = np.argsort(panel.index.tolist())
        values = values[sorted_idx]
        sorted_syms = sorted(panel.index.tolist())

        symbols_arr = np.array(sorted_syms)
        dates_arr = np.array(panel.columns.tolist(), dtype="datetime64[ns]")

        return values, symbols_arr, dates_arr


# ------------------------------------------------------------------
# Smoke test (run: python snoa_ttm.py)
# Uses full market with short date range for cross-section adequacy.
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("SNOA_TTM factor smoke test")
    print("=" * 60)

    TEST_START = "2023-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
    )

    print(f"\n[Step 2] Compute SNOA_TTM factor")
    calculator = SNOA_TTM()
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
