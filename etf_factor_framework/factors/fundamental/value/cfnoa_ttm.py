"""
CFNOA_TTM factor (Operating Cash Flow to Net Operating Assets Market Value TTM)

Formula: OCF_TTM / NOA_Market
  - OCF_TTM: operating cash flow TTM (yuan, from financial_statements)
      Primary:  q_cfs_ncffoa_ttm  (lixinger pre-computed TTM, ~95.6% coverage)
      Fallback: q_cfs_ncffoa_c    (cumulative per-quarter, TTM-rolled via 4-quarter sum)
  - NOA_Market = mc - fin_net_assets  (yuan, from fundamental)
      mc            = lixinger.fundamental daily total market cap (yuan)
      fin_net_assets = fnpa if available, else fpa - fb
      fnpa          = lixinger.fundamental financial net assets (yuan)
      fpa           = lixinger.fundamental financial assets (yuan, fallback)
      fb            = lixinger.fundamental financial liabilities (yuan, fallback)

Note on units: lixinger.fundamental stores mc/fnpa/fpa/fb in yuan (CNY),
not in 亿元 as sometimes documented. financial_statements fields are also
in yuan, so no unit conversion is needed — the ratio is dimensionless.

Factor direction: positive (higher CFNOA_TTM = cheaper on operating cash flow / NOA basis)
Factor category: value - denominator-improved value
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from core.factor_data import FactorData
from factors.fundamental.fundamental_data import FundamentalData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator

FACTOR_NAME = "CFNOA_TTM"
FACTOR_DIRECTION = 1  # positive: higher CFNOA_TTM (cheaper on OCF/NOA basis) is better

# Financial industry sectors: NOA concept does not apply
FINANCIAL_INDUSTRIES = {"银行", "非银金融"}


class CFNOA_TTM(FundamentalFactorCalculator):
    """
    Operating Cash Flow to Net Operating Assets Market Value TTM factor.

    CFNOA_TTM = OCF_TTM / NOA_Market
    where:
      OCF_TTM    = operating cash flow TTM (yuan)
                   primary:  q_cfs_ncffoa_ttm (lixinger pre-computed TTM)
                   fallback: q_cfs_ncffoa_c   (TTM-rolled via 4-quarter sum)
      NOA_Market = mc - fnpa  (yuan, daily; fnpa falls back to fpa - fb if NaN)

    Financial industry stocks (银行/非银金融) are set to NaN.
    NOA_Market <= 0 is set to NaN (can't compute ratio for non-positive denominator).
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
        Compute CFNOA_TTM daily panel (N stocks x T days).

        Returns:
            FactorData with values = CFNOA_TTM (dimensionless ratio)
        """
        # --- 1. Load daily valuation panels from lixinger.fundamental ---
        mc_v, mc_s, mc_d = fundamental_data.get_valuation_panel("mc")
        fnpa_v, fnpa_s, fnpa_d = fundamental_data.get_valuation_panel("fnpa")
        fpa_v, fpa_s, fpa_d = fundamental_data.get_valuation_panel("fpa")
        fb_v, fb_s, fb_d = fundamental_data.get_valuation_panel("fb")

        if mc_v.size == 0:
            raise ValueError("CFNOA_TTM: get_valuation_panel('mc') returned empty array")

        # --- 2. Load OCF TTM primary (lixinger pre-computed field) ---
        ocf_v, ocf_s, ocf_d = fundamental_data.get_daily_panel("q_cfs_ncffoa_ttm")

        # --- 3. Compute OCF TTM fallback from cumulative quarterly field ---
        # Use mc_d as target_dates so panel shares the same trading calendar
        ocf_fb_v, ocf_fb_s, ocf_fb_d = self._compute_cumulative_ttm_panel(
            fundamental_data, "q_cfs_ncffoa_c", mc_d
        )

        # --- 4. Common symbols: intersection of mc with union of primary/fallback ---
        all_ocf_syms = np.array([], dtype=object)
        if ocf_v.size > 0:
            all_ocf_syms = np.union1d(all_ocf_syms, ocf_s)
        if ocf_fb_v.size > 0:
            all_ocf_syms = np.union1d(all_ocf_syms, ocf_fb_s)

        common_syms = np.intersect1d(mc_s, all_ocf_syms)
        if len(common_syms) == 0:
            raise ValueError("CFNOA_TTM: no common symbols across mc/ocf panels")

        # --- 5. Common dates (all panels aligned to same trading calendar) ---
        common_dates = mc_d

        N = len(common_syms)
        T = len(common_dates)

        def _sym_idx(arr, targets):
            """Map each target symbol to its index in arr (arr must be sorted)."""
            return np.searchsorted(arr, targets)

        def _date_idx(arr, targets):
            """Map each target date to its index in arr (arr must be sorted)."""
            return np.searchsorted(arr, targets)

        # --- 6. Build OCF panel (N, T): primary first, fill NaN from fallback ---
        ocf_aligned = np.full((N, T), np.nan)

        if ocf_v.size > 0:
            primary_common = np.intersect1d(ocf_s, common_syms)
            if len(primary_common) > 0:
                src_si = _sym_idx(ocf_s, primary_common)
                dst_si = _sym_idx(common_syms, primary_common)
                di = _date_idx(ocf_d, common_dates)
                ocf_aligned[dst_si] = ocf_v[src_si][:, di]

        if ocf_fb_v.size > 0:
            fb_common = np.intersect1d(ocf_fb_s, common_syms)
            if len(fb_common) > 0:
                src_si = _sym_idx(ocf_fb_s, fb_common)
                dst_si = _sym_idx(common_syms, fb_common)
                di = _date_idx(ocf_fb_d, common_dates)
                fb_data = ocf_fb_v[src_si][:, di]
                still_nan = np.isnan(ocf_aligned[dst_si])
                valid_fb = ~np.isnan(fb_data)
                use_fb = still_nan & valid_fb
                ocf_aligned[dst_si] = np.where(use_fb, fb_data, ocf_aligned[dst_si])

        # --- 7. Align mc to common_syms / common_dates ---
        mc_si = _sym_idx(mc_s, common_syms)
        mc_di = _date_idx(mc_d, common_dates)
        mc_aligned = mc_v[mc_si][:, mc_di]          # (N, T), yuan

        # --- 8. Build financial net assets panel for common_syms / common_dates ---
        fin_net_assets = np.full((N, T), np.nan)

        # Priority 1: fnpa
        if fnpa_v.size > 0:
            fnpa_common = np.intersect1d(fnpa_s, common_syms)
            if len(fnpa_common) > 0:
                fnpa_src_idx = _sym_idx(fnpa_s, fnpa_common)
                fnpa_dst_idx = _sym_idx(common_syms, fnpa_common)
                fnpa_di = _date_idx(fnpa_d, common_dates)
                fin_net_assets[fnpa_dst_idx] = fnpa_v[fnpa_src_idx][:, fnpa_di]

        # Priority 2: fpa - fb where fnpa is still NaN
        if fpa_v.size > 0 and fb_v.size > 0:
            fpa_fb_common = np.intersect1d(np.intersect1d(fpa_s, fb_s), common_syms)
            if len(fpa_fb_common) > 0:
                fpa_src_idx = _sym_idx(fpa_s, fpa_fb_common)
                fb_src_idx = _sym_idx(fb_s, fpa_fb_common)
                dst_idx = _sym_idx(common_syms, fpa_fb_common)
                fpa_di = _date_idx(fpa_d, common_dates)
                fb_di = _date_idx(fb_d, common_dates)
                fpa_aligned = fpa_v[fpa_src_idx][:, fpa_di]
                fb_aligned = fb_v[fb_src_idx][:, fb_di]
                fallback_fna = fpa_aligned - fb_aligned
                still_nan = np.isnan(fin_net_assets[dst_idx])
                valid_fallback = ~np.isnan(fpa_aligned) & ~np.isnan(fb_aligned)
                use_fallback = still_nan & valid_fallback
                fin_net_assets[dst_idx] = np.where(
                    use_fallback, fallback_fna, fin_net_assets[dst_idx]
                )

        # --- 9. NOA_market = mc - fin_net_assets (yuan) ---
        noa_market = mc_aligned - fin_net_assets

        # --- 10. Compute CFNOA_TTM (dimensionless, both in yuan) ---
        with np.errstate(divide="ignore", invalid="ignore"):
            cfnoa = np.where(
                (noa_market > 0)
                & ~np.isnan(ocf_aligned)
                & ~np.isnan(fin_net_assets),
                ocf_aligned / noa_market,
                np.nan,
            )
        cfnoa = cfnoa.astype(np.float64)

        # --- 11. Financial industry: force NaN ---
        industry_map = fundamental_data.get_industry_map()
        for i, sym in enumerate(common_syms):
            if industry_map.get(sym) in FINANCIAL_INDUSTRIES:
                cfnoa[i, :] = np.nan

        nan_ratio = np.isnan(cfnoa).mean()
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(
                f"CFNOA_TTM NaN ratio is high: {nan_ratio:.1%}, please check data"
            )

        return FactorData(
            values=cfnoa,
            symbols=common_syms,
            dates=common_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cumulative_ttm_panel(
        self,
        fundamental_data: FundamentalData,
        field: str,
        target_dates: np.ndarray,
    ):
        """
        Build a daily panel of TTM values from lixinger single-quarter field values.

        Note: Despite the field name suffix "_c" (累计), lixinger.financial_statements
        stores SINGLE-QUARTER values (not year-to-date cumulative). The June 30 row
        stores Q2-only value, the September 30 row stores Q3-only value, etc.
        TTM is therefore computed as the rolling sum of the 4 most recently
        disclosed quarters (leak-free: ordered by report_date).

        TTM rolling formula:
          ttm(t) = sum of field values across the 4 most recently disclosed
                   quarterly reports with report_date <= t

        Forward-fills the computed TTM using report_date (disclosure date),
        preventing future data leakage.

        Args:
            fundamental_data: FundamentalData instance (raw data already loaded)
            field: column name in financial_statements (e.g. 'q_cfs_ncffoa_c')
            target_dates: trading dates array (datetime64[ns]) to project onto

        Returns:
            (values, symbols, dates) tuple like get_daily_panel()
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

        # Extract records with non-null field value
        df = raw[["stock_code", "report_date", field]].dropna(
            subset=[field]
        ).copy()

        if df.empty:
            return empty

        # Ensure report_date is tz-naive, filter out future data
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)
        df = df[df["report_date"] <= fundamental_data.end_date]

        if df.empty:
            return empty

        # Sort by (stock_code, report_date) ascending so rolling is in disclosure order
        df = df.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

        # Rolling sum of the 4 most recently disclosed quarters per stock
        # min_periods=4: require at least 4 quarters of history; NaN otherwise
        df["ttm_val"] = (
            df.groupby("stock_code")[field]
            .transform(lambda x: x.rolling(4, min_periods=4).sum())
        )

        # Drop records where TTM could not be computed (fewer than 4 quarters available)
        ttm_df = df[["stock_code", "report_date", "ttm_val"]].dropna(subset=["ttm_val"])

        if ttm_df.empty:
            return empty

        ttm_df["report_date"] = pd.to_datetime(ttm_df["report_date"])

        # Map stock_code -> symbol (DuckDB format)
        symbol_map = (
            raw[["stock_code", "symbol"]]
            .drop_duplicates("stock_code")
            .set_index("stock_code")["symbol"]
        )
        ttm_df["symbol"] = ttm_df["stock_code"].map(symbol_map)
        ttm_df = ttm_df.dropna(subset=["symbol"])

        if ttm_df.empty:
            return empty

        # Pivot: rows=symbol, cols=report_date, values=ttm_val
        pivot = ttm_df.pivot_table(
            index="symbol",
            columns="report_date",
            values="ttm_val",
            aggfunc="last",
        )

        # Forward-fill to trading dates (same logic as get_daily_panel)
        all_dates = pivot.columns.union(trading_dates).sort_values()
        pivot = pivot.reindex(columns=all_dates)
        panel = pivot.ffill(axis=1).reindex(columns=trading_dates)

        values = panel.values.astype(np.float64)

        # Sort symbols for consistent ordering
        sorted_syms = sorted(panel.index.tolist())
        sorted_idx = np.argsort(panel.index.tolist())
        values = values[sorted_idx]

        symbols_arr = np.array(sorted_syms)
        dates_arr = np.array(panel.columns.tolist(), dtype="datetime64[ns]")

        return values, symbols_arr, dates_arr


# ------------------------------------------------------------------
# Smoke test (run: python cfnoa_ttm.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("CFNOA_TTM factor smoke test")
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

    # q_cfs_ncffoa_ttm (primary OCF TTM, pre-computed)
    ocf_v, ocf_s, ocf_d = fd.get_daily_panel("q_cfs_ncffoa_ttm")
    assert ocf_v.size > 0, "q_cfs_ncffoa_ttm returned empty array"
    nr_ocf = np.isnan(ocf_v).mean()
    print(f"  q_cfs_ncffoa_ttm shape={ocf_v.shape}, NaN={nr_ocf:.1%}")

    # q_cfs_ncffoa_c (fallback OCF cumulative)
    ocf_c_v, ocf_c_s, ocf_c_d = fd.get_daily_panel("q_cfs_ncffoa_c")
    assert ocf_c_v.size > 0, "q_cfs_ncffoa_c returned empty array"
    nr_ocfc = np.isnan(ocf_c_v).mean()
    print(f"  q_cfs_ncffoa_c shape={ocf_c_v.shape}, NaN={nr_ocfc:.1%}")

    # mc
    mc_v, mc_s, mc_d = fd.get_valuation_panel("mc")
    assert mc_v.size > 0, "mc returned empty array"
    nr_mc = np.isnan(mc_v).mean()
    assert nr_mc < 0.5, f"mc NaN ratio too high: {nr_mc:.1%}"
    print(f"  mc shape={mc_v.shape}, NaN={nr_mc:.1%}, sample={mc_v[:, -1]}")

    # fnpa (allow high NaN: ~50% global coverage)
    fnpa_v, fnpa_s, fnpa_d = fd.get_valuation_panel("fnpa")
    print(f"  fnpa shape={fnpa_v.shape}, NaN={np.isnan(fnpa_v).mean():.1%} "
          "(high NaN OK, ~50% global coverage expected)")

    # fpa, fb (fallback for fnpa)
    fpa_v, fpa_s, fpa_d = fd.get_valuation_panel("fpa")
    fb_v, fb_s, fb_d = fd.get_valuation_panel("fb")
    print(f"  fpa shape={fpa_v.shape}, NaN={np.isnan(fpa_v).mean():.1%}")
    print(f"  fb shape={fb_v.shape}, NaN={np.isnan(fb_v).mean():.1%}")

    print("[PASS] Data validation")

    print(f"\n[Step 3] Compute CFNOA_TTM factor")
    calculator = CFNOA_TTM()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ {pd.Timestamp(result.dates[-1]).date()}")

    # --- Section 4.1 smoke test assertions ---

    # Shape & dtype
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"

    # NaN boundary
    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\n[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

    # --- Additional diagnostics ---
    print(f"\nSample values (last 5 dates) per stock:")
    for i, sym in enumerate(result.symbols):
        row = result.values[i]
        last5 = row[-5:]
        print(f"  {sym}: {last5}")

    print(f"\nLast cross-section stats:")
    last_cs = result.values[:, -1]
    valid_cs = last_cs[~np.isnan(last_cs)]
    if len(valid_cs):
        print(f"  N valid: {len(valid_cs)}")
        print(f"  mean:    {valid_cs.mean():.6f}")
        print(f"  median:  {np.median(valid_cs):.6f}")
        print(f"  min:     {valid_cs.min():.6f}")
        print(f"  max:     {valid_cs.max():.6f}")

    print(f"\n[NOTE] Leakage detection is handled in the next step.")
