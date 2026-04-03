"""
Quality Composite Score (QCS) factor

Composite quality factor inspired by Asness, Frazzini & Pedersen (2019)
"Quality Minus Junk" (QMJ). Combines 4 quality dimensions:

  1. Profitability  (z_prof):  GPM, ROA, ROIC, CF_ROA, OP_margin
  2. Accounting Quality (z_acct): -ACCRUALS, CF_NI_ratio
  3. Payout         (z_payout): FCF_yield_on_assets, -Net_financing
  4. Investment     (z_inv):   -ASSET_GR, -CAPEX_GR

  QUALITY = (z_prof + z_acct + z_payout + z_inv) / 4

Factor direction: positive (higher composite quality -> higher expected return)
Factor category: quality - composite

Data source: lixinger financial_statements
Fields used:
  q_ps_gp_m_t, q_m_roa_t, q_m_roic_t, q_cfs_ncffoa_ttm, q_ps_op_s_r_t,
  q_m_ncffoa_np_r_t, q_m_fcf_ttm, q_cfs_ncfffa_ttm, q_bs_ta_t,
  q_bs_ta_t_y2y, q_cfs_cpfpfiaolta_c

Notes:
  - Financial sector stocks (banks, non-bank financials) are excluded (NaN).
  - ROIC is expected to have ~50% coverage; dimension z-score uses available sub-indicators.
  - CAPEX TTM is computed from rolling 4-quarter sum of q_cfs_cpfpfiaolta_c.
  - Industry-neutral z-scoring uses get_industry_map() (Lixinger industry classification).
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from core.factor_data import FactorData
from factors.fundamental.fundamental_calculator import FundamentalFactorCalculator
from factors.fundamental.fundamental_data import FundamentalData

FACTOR_NAME = "quality_QCS"
FACTOR_DIRECTION = 1  # positive

# Financial sectors to exclude (Lixinger industry names)
_EXCLUDE_INDUSTRIES = {"银行", "非银金融"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_mainboard(symbol: str) -> bool:
    """True if symbol is A-share mainboard (60xxxx or 00xxxx)."""
    m = re.search(r"(\d{6})", symbol)
    if not m:
        return False
    code = m.group(1)
    return code.startswith("60") or code.startswith("00")


def _winsorize_df(df: pd.DataFrame, n_sigma: float = 3.0) -> pd.DataFrame:
    """Cross-sectional 3-sigma winsorize (vectorised over all dates)."""
    mu = df.mean(axis=0)
    sigma = df.std(axis=0)
    lo = mu - n_sigma * sigma
    hi = mu + n_sigma * sigma
    clipped = df.clip(lower=lo, upper=hi, axis=1)
    clipped[df.isna()] = np.nan
    return clipped


def _industry_zscore_df(
    df: pd.DataFrame,
    industry_map: dict,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """
    Industry-neutral cross-sectional z-score.

    For each industry and each date, standardise values within the group.
    Fall back to global z-score for groups with fewer than min_group_size
    valid stocks on a given date.
    """
    industries = pd.Series(
        {sym: industry_map.get(sym, "__unknown__") for sym in df.index},
        name="industry",
    )

    # Global z-score as fallback
    global_mu = df.mean(axis=0)
    global_std = df.std(axis=0).replace(0.0, np.nan)
    global_z = (df - global_mu) / global_std

    result = pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)

    for ind_name, idx in industries.groupby(industries).groups.items():
        sub = df.loc[idx]
        mu = sub.mean(axis=0)
        std = sub.std(axis=0).replace(0.0, np.nan)
        ind_z = (sub - mu) / std

        n_valid = sub.notna().sum(axis=0)
        use_ind = n_valid >= min_group_size

        combined = ind_z.copy()
        if (~use_ind).any():
            combined.loc[:, ~use_ind] = global_z.loc[idx, ~use_ind]

        result.loc[idx] = combined

    return result


def _panel_to_df(values: np.ndarray, symbols: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=symbols, columns=pd.DatetimeIndex(dates))


def _align_to_common(dfs: dict) -> dict:
    """Reindex all DataFrames in dfs to a common (union) symbol set."""
    all_symbols = sorted(set().union(*[df.index.tolist() for df in dfs.values()]))
    return {k: df.reindex(all_symbols) for k, df in dfs.items()}


# ---------------------------------------------------------------------------
# Factor class
# ---------------------------------------------------------------------------


class quality_QCS(FundamentalFactorCalculator):
    """
    Composite Quality Score factor (QCS).

    Four-dimension equal-weight composite:
      z_prof + z_acct + z_payout + z_inv, each from industry-neutral z-scores.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        fd = fundamental_data

        # ---- 1. Load raw field panels ----
        dfs = self._load_all_panels(fd)

        # ---- 2. Build sub-indicators (DataFrames, same symbol × date axes) ----
        dfs = _align_to_common(dfs)
        dates = list(dfs.values())[0].columns
        symbols = list(dfs.values())[0].index

        # Mainboard filter
        mb_mask = np.array([_is_mainboard(s) for s in symbols])
        symbols_mb = symbols[mb_mask]

        # Get industry map for industry-neutral z-score & financial exclusion
        industry_map = fd.get_industry_map()

        # Identify financial sector symbols to exclude
        fin_syms = {
            sym for sym, ind in industry_map.items() if ind in _EXCLUDE_INDUSTRIES
        }

        # Helper: restrict panel to mainboard, set financial stocks to NaN
        def prep(df: pd.DataFrame) -> pd.DataFrame:
            d = df.loc[symbols_mb].copy()
            fin_in_mb = [s for s in d.index if s in fin_syms]
            if fin_in_mb:
                d.loc[fin_in_mb] = np.nan
            return d

        # ---- 3. Compute sub-indicators ----
        ta = prep(dfs["ta"])

        # -- Profitability --
        gpm = prep(dfs["gpm"])
        roa = prep(dfs["roa"])
        roic = prep(dfs["roic"])  # may be ~50% NaN; handled in dim average
        cfoa = prep(dfs["ncffoa"])
        op = prep(dfs["opmargin"])

        # CF_ROA = operating CF / total assets
        cf_roa = cfoa / ta.replace(0, np.nan)

        # -- Accounting Quality --
        # ACCRUALS = ROA - CF_ROA; negate so lower accruals = better quality
        neg_accruals = -(roa - cf_roa)
        cfni = prep(dfs["cfni"])  # CF/NI ratio (already positive = better)

        # -- Payout --
        fcf = prep(dfs["fcf"])
        ncfffa = prep(dfs["ncfffa"])

        fcf_yield = fcf / ta.replace(0, np.nan)
        neg_net_fin = -ncfffa / ta.replace(0, np.nan)

        # -- Investment --
        neg_tagr = -prep(dfs["tagr"])  # negate asset growth

        if dfs.get("neg_capex_gr") is not None:
            neg_capex_gr = prep(dfs["neg_capex_gr"])
        else:
            neg_capex_gr = pd.DataFrame(np.nan, index=symbols_mb, columns=dates)

        # ---- 4. Industry-neutral z-score per sub-indicator ----
        def ind_z(df: pd.DataFrame) -> pd.DataFrame:
            w = _winsorize_df(df)
            return _industry_zscore_df(w, industry_map)

        z_gpm = ind_z(gpm)
        z_roa = ind_z(roa)
        z_roic = ind_z(roic)
        z_cf_roa = ind_z(cf_roa)
        z_op = ind_z(op)

        z_neg_accruals = ind_z(neg_accruals)
        z_cfni = ind_z(cfni)

        z_fcf_yield = ind_z(fcf_yield)
        z_neg_fin = ind_z(neg_net_fin)

        z_neg_tagr = ind_z(neg_tagr)
        z_neg_capex = ind_z(neg_capex_gr)

        # ---- 5. Dimension scores (NaN-safe mean of sub-indicators) ----
        def dim_mean(*zs: pd.DataFrame, min_valid: int = 1) -> pd.DataFrame:
            """Mean of sub-indicators, requiring at least min_valid non-NaN."""
            stacked = np.stack([z.values for z in zs], axis=0)  # (K, N, T)
            n_valid = np.sum(~np.isnan(stacked), axis=0)        # (N, T)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(stacked, axis=0)
            mean[n_valid < min_valid] = np.nan
            return pd.DataFrame(mean, index=zs[0].index, columns=zs[0].columns)

        # Profitability: 5 sub-indicators (roic may be missing, min_valid=3)
        z_prof = dim_mean(z_gpm, z_roa, z_roic, z_cf_roa, z_op, min_valid=3)

        # Accounting quality: 2 sub-indicators
        z_acct = dim_mean(z_neg_accruals, z_cfni, min_valid=1)

        # Payout: 2 sub-indicators
        z_payout = dim_mean(z_fcf_yield, z_neg_fin, min_valid=1)

        # Investment: 2 sub-indicators (capex may be unavailable early on)
        z_inv = dim_mean(z_neg_tagr, z_neg_capex, min_valid=1)

        # ---- 6. Composite QUALITY = equal-weight average of 4 dimensions ----
        quality_raw = dim_mean(z_prof, z_acct, z_payout, z_inv, min_valid=3)

        # Final global cross-sectional z-score
        q_mu = quality_raw.mean(axis=0)
        q_std = quality_raw.std(axis=0).replace(0.0, np.nan)
        quality_final = (quality_raw - q_mu) / q_std

        # ---- 7. Pack into FactorData ----
        values_out = quality_final.values.astype(np.float64)
        symbols_out = np.array(quality_final.index.tolist())
        dates_out = np.array(quality_final.columns, dtype="datetime64[ns]")

        return FactorData(
            values=values_out,
            symbols=symbols_out,
            dates=dates_out,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_panels(self, fd: FundamentalData) -> dict:
        """Load all required field panels; returns dict of DataFrames."""

        def to_df(field: str) -> pd.DataFrame:
            v, s, d = fd.get_daily_panel(field)
            return _panel_to_df(v, s, d)

        dfs = {
            "gpm": to_df("q_ps_gp_m_t"),
            "roa": to_df("q_m_roa_t"),
            "roic": to_df("q_m_roic_t"),
            "ncffoa": to_df("q_cfs_ncffoa_ttm"),
            "opmargin": to_df("q_ps_op_s_r_t"),
            "ta": to_df("q_bs_ta_t"),
            "cfni": to_df("q_m_ncffoa_np_r_t"),
            "fcf": to_df("q_m_fcf_ttm"),
            "ncfffa": to_df("q_cfs_ncfffa_ttm"),
            "tagr": to_df("q_bs_ta_t_y2y"),
        }

        # CAPEX growth requires raw quarterly data
        capex_result = self._compute_neg_capex_gr(fd)
        if capex_result is not None:
            dfs["neg_capex_gr"] = capex_result
        else:
            # Use the date/symbol axes from an already-loaded panel as placeholder
            ref_df = dfs["tagr"]
            dfs["neg_capex_gr"] = pd.DataFrame(
                np.nan, index=ref_df.index, columns=ref_df.columns
            )

        return dfs

    def _compute_neg_capex_gr(self, fd: FundamentalData) -> pd.DataFrame | None:
        """
        Compute -CAPEX_GR daily panel from raw quarterly q_cfs_cpfpfiaolta_c.

        CAPEX_TTM  = rolling 4-quarter sum of q_cfs_cpfpfiaolta_c (single quarter)
        CAPEX_GR   = (CAPEX_TTM - CAPEX_TTM.shift(4)) / |CAPEX_TTM.shift(4)|
        Returns -CAPEX_GR (lower capex growth -> better quality -> positive direction)
        """
        # Ensure raw data is populated
        fd._load_raw_data()
        raw = fd._raw_data

        if raw is None or raw.empty:
            return None
        if "q_cfs_cpfpfiaolta_c" not in raw.columns:
            return None

        df = (
            raw[["symbol", "report_date", "q_cfs_cpfpfiaolta_c"]]
            .dropna(subset=["q_cfs_cpfpfiaolta_c", "report_date"])
            .copy()
        )
        if df.empty:
            return None

        df = df.sort_values(["symbol", "report_date"]).reset_index(drop=True)

        # TTM = rolling 4-quarter sum per stock
        df["capex_ttm"] = df.groupby("symbol")["q_cfs_cpfpfiaolta_c"].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

        # Prior-year TTM = shift 4 rows within each stock (same quarter last year)
        df["capex_ttm_1y"] = df.groupby("symbol")["capex_ttm"].shift(4)

        # YoY growth (avoid division by near-zero)
        valid_denom = df["capex_ttm_1y"].abs() > 1e-6
        df["capex_gr"] = np.nan
        df.loc[valid_denom, "capex_gr"] = (
            (df.loc[valid_denom, "capex_ttm"] - df.loc[valid_denom, "capex_ttm_1y"])
            / df.loc[valid_denom, "capex_ttm_1y"].abs()
        )

        df["-capex_gr"] = -df["capex_gr"]
        df = df.dropna(subset=["-capex_gr"])

        if df.empty:
            return None

        # Prevent future data leakage: only use report_date <= fd.end_date
        df = df[df["report_date"] <= fd.end_date]
        if df.empty:
            return None

        # Forward-fill derived value using report_date as signal date
        trading_dates = fd.trading_dates
        pivot = df.pivot_table(
            index="symbol",
            columns="report_date",
            values="-capex_gr",
            aggfunc="last",
        )

        all_dates = pivot.columns.union(trading_dates).sort_values()
        panel = pivot.reindex(columns=all_dates).ffill(axis=1)
        panel = panel.reindex(columns=trading_dates)

        return panel.astype(float)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("quality_QCS factor smoke test")
    print("=" * 60)

    TEST_START = "2020-01-01"
    TEST_END = "2024-12-31"
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print(f"\n[Step 2] Compute quality_QCS factor")
    calculator = quality_QCS()
    result = calculator.calculate(fd)

    print(f"\nFactor shape: {result.values.shape}")
    print(f"Symbols: {result.symbols}")
    print(f"Date range: {pd.Timestamp(result.dates[0]).date()} ~ {pd.Timestamp(result.dates[-1]).date()}")

    # --- Smoke test checks ---
    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    assert nan_ratio < 0.8, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    assert not np.isinf(valid).any(), "Factor contains inf values"

    # Idempotency
    result2 = calculator.calculate(fd)
    both_nan = np.isnan(result.values) & np.isnan(result2.values)
    assert np.all((result.values == result2.values) | both_nan), "Idempotency failed"

    print(f"\n[PASS] Smoke test: shape={result.values.shape}, NaN={nan_ratio:.1%}")

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
        print(f"  mean: {valid_cs.mean():.4f}")
        print(f"  std:  {valid_cs.std():.4f}")
        print(f"  min:  {valid_cs.min():.4f}")
        print(f"  max:  {valid_cs.max():.4f}")

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
