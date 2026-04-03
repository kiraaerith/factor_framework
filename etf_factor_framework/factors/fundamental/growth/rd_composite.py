"""
RD_COMPOSITE factor (Comprehensive R&D Capability Composite)

Composite factor synthesizing 5 sub-factors that measure a company's R&D strength
from multiple angles: funding intensity, brand investment, and technology assets.

Sub-factors (5 active; 2 omitted due to missing data):
  - RD_TO_MV    : R&D expense TTM / market cap (funding intensity, value dimension)
  - AD_TO_MV    : Selling expense TTM / market cap (brand investment proxy)
  - RD_TO_NP    : R&D expense TTM / net profit (funding intensity, profit dimension)
  - RD_TO_REV   : R&D expense TTM / revenue (funding intensity, scale dimension)
  - PATENT_TO_MV: Intangible assets / market cap (technology asset stock)

Omitted sub-factors (data unavailable in lixinger.db):
  - RD_EMP_RATE : R&D headcount ratio (from annual report footnotes, not structured)
  - PHD_RATE    : PhD employee ratio (from annual report, not structured)

Synthesis:
  1. Each sub-factor is individually standardized:
     winsorize (3sigma) -> zscore -> size+industry neutralize -> winsorize -> zscore
  2. Equal-weight nanmean across available sub-factors.
     A stock needs >= 2 valid sub-factors; otherwise NaN.
  3. Composite post-processed:
     winsorize (3sigma) -> zscore -> size+industry neutralize -> winsorize -> zscore

Factor direction: positive
Factor category: growth - comprehensive R&D capability
"""

import os
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

from factors.fundamental.growth.rd_to_mv import RD_TO_MV
from factors.fundamental.growth.ad_to_mv import AD_TO_MV
from factors.fundamental.growth.rd_to_np import RD_TO_NP
from factors.fundamental.growth.rd_to_rev import RD_TO_REV
from factors.fundamental.growth.patent_to_mv import PATENT_TO_MV

FACTOR_NAME = "RD_COMPOSITE"
FACTOR_DIRECTION = 1  # positive: higher composite R&D capability is better
MIN_VALID_SUBFACTORS = 2  # minimum valid sub-factors per stock-date to compute composite


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

def _align_panel(values: np.ndarray, symbols: np.ndarray, target_symbols: np.ndarray) -> np.ndarray:
    """
    Align panel (N, T) to target_symbols (N', T); missing symbols filled with NaN.
    Both symbols and target_symbols must be sorted (or at least consistent).
    """
    N_target = len(target_symbols)
    T = values.shape[1] if values.ndim == 2 else 0
    result = np.full((N_target, T), np.nan, dtype=np.float64)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    for i, sym in enumerate(target_symbols):
        j = sym_idx.get(sym)
        if j is not None:
            result[i] = values[j]
    return result


def _winsorize_3sigma(arr: np.ndarray) -> np.ndarray:
    """Per-column 3-sigma winsorize for a (N, T) array."""
    result = arr.copy()
    T = arr.shape[1]
    for t in range(T):
        col = arr[:, t]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() < 3:
            continue
        mu = col[valid_mask].mean()
        sigma = col[valid_mask].std(ddof=1)
        if sigma < 1e-10:
            continue
        lo, hi = mu - 3 * sigma, mu + 3 * sigma
        clipped = np.clip(col, lo, hi)
        clipped[~valid_mask] = np.nan
        result[:, t] = clipped
    return result


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Per-column cross-sectional z-score for a (N, T) array."""
    result = arr.copy()
    T = arr.shape[1]
    for t in range(T):
        col = arr[:, t]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() < 3:
            continue
        mu = col[valid_mask].mean()
        sigma = col[valid_mask].std(ddof=1)
        if sigma < 1e-10:
            result[:, t] = np.where(valid_mask, 0.0, np.nan)
        else:
            result[:, t] = np.where(valid_mask, (col - mu) / sigma, np.nan)
    return result


def _neutralize_size_industry(
    arr: np.ndarray,
    symbols: np.ndarray,
    mc_arr: np.ndarray,
    industry_map: dict,
) -> np.ndarray:
    """
    Per-date OLS regression with log(mc) and industry dummies as controls.
    Residuals are returned as the neutralized factor values.
    Requires >= 10 valid observations per cross-section; otherwise returns original.
    """
    N, T = arr.shape
    result = arr.copy()
    log_mc = np.log(np.where(mc_arr > 0, mc_arr, np.nan))  # (N, T)

    industries = np.array([industry_map.get(s) for s in symbols], dtype=object)

    for t in range(T):
        y = arr[:, t]
        mc = log_mc[:, t]
        ind = industries

        valid = (
            ~np.isnan(y)
            & ~np.isnan(mc)
            & np.array([v is not None for v in ind])
        )
        if valid.sum() < 10:
            continue

        y_v = y[valid]
        mc_v = mc[valid]
        ind_v = ind[valid].astype(str)

        unique_ind = np.unique(ind_v)
        if len(unique_ind) >= 2:
            # Industry dummies (drop first category for identification)
            ind_ref = unique_ind[0]
            ind_cols = unique_ind[1:]
            dummies = np.zeros((len(y_v), len(ind_cols)), dtype=np.float64)
            for j, ind_name in enumerate(ind_cols):
                dummies[:, j] = (ind_v == ind_name).astype(np.float64)
            X = np.column_stack([np.ones(len(y_v)), mc_v, dummies])
        else:
            X = np.column_stack([np.ones(len(y_v)), mc_v])

        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_v, rcond=None)
            residual = y_v - X @ coef
            result[valid, t] = residual
        except Exception:
            pass  # retain original values on failure

    return result


def _standardize_pipeline(
    arr: np.ndarray,
    symbols: np.ndarray,
    mc_arr: np.ndarray,
    industry_map: dict,
) -> np.ndarray:
    """
    Full 4-step standardization:
    winsorize(3sigma) -> zscore -> size+industry neutralize -> winsorize -> zscore
    """
    arr = _winsorize_3sigma(arr)
    arr = _zscore(arr)
    arr = _neutralize_size_industry(arr, symbols, mc_arr, industry_map)
    arr = _winsorize_3sigma(arr)
    arr = _zscore(arr)
    return arr


# ------------------------------------------------------------------
# Factor class
# ------------------------------------------------------------------

class RD_COMPOSITE(FundamentalFactorCalculator):
    """
    Comprehensive R&D Capability Composite factor.

    Synthesizes 5 sub-factors (RD_TO_MV, AD_TO_MV, RD_TO_NP, RD_TO_REV,
    PATENT_TO_MV) into an equal-weight composite after cross-sectional
    standardization and neutralization of each sub-factor.

    Minimum 2 valid sub-factors required per stock-date; otherwise NaN.
    Universe: A-share mainboard (inherited from sub-factors: 60xxxx / 00xxxx).
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "direction": FACTOR_DIRECTION,
            "sub_factors": [
                "RD_TO_MV", "AD_TO_MV", "RD_TO_NP", "RD_TO_REV", "PATENT_TO_MV",
            ],
            "min_valid_subfactors": MIN_VALID_SUBFACTORS,
        }

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        Compute RD_COMPOSITE daily panel (N stocks x T days).

        Steps:
          1. Compute 5 sub-factor panels via their respective calculators.
          2. Build union symbol set; align all panels to (union_symbols, dates).
          3. Fetch market cap panel and industry map for neutralization.
          4. Standardize each sub-factor: winsorize -> zscore -> neutralize -> winsorize -> zscore.
          5. Equal-weight nanmean (min MIN_VALID_SUBFACTORS valid per cell).
          6. Post-process composite: same 4-step standardization.

        Returns:
            FactorData: N stocks x T days
        """
        # --- Step 1: Compute all sub-factor panels ---
        sub_calculators = [
            ("RD_TO_MV",    RD_TO_MV()),
            ("AD_TO_MV",    AD_TO_MV()),
            ("RD_TO_NP",    RD_TO_NP()),
            ("RD_TO_REV",   RD_TO_REV()),
            ("PATENT_TO_MV", PATENT_TO_MV()),
        ]

        sub_results = []
        for label, calc in sub_calculators:
            fd_result = calc.calculate(fundamental_data)
            sub_results.append((label, fd_result))

        if not sub_results:
            raise ValueError(f"{FACTOR_NAME}: no sub-factor results produced")

        # --- Step 2: Build union symbol set ---
        all_symbols_set = set()
        for _, fd_result in sub_results:
            all_symbols_set.update(fd_result.symbols.tolist())
        all_symbols = np.array(sorted(all_symbols_set))

        # Use dates from first sub-factor (all share the same trading calendar)
        ref_dates = sub_results[0][1].dates
        T = len(ref_dates)
        N = len(all_symbols)

        # --- Step 3: Fetch market cap panel and industry map for neutralization ---
        mc_v, mc_s, _ = fundamental_data.get_market_cap_panel()
        mc_aligned = _align_panel(mc_v, mc_s, all_symbols)  # (N, T)

        industry_map = fundamental_data.get_industry_map()

        # --- Step 4: Align and standardize each sub-factor ---
        std_panels = []
        for label, fd_result in sub_results:
            raw = _align_panel(fd_result.values, fd_result.symbols, all_symbols)  # (N, T)
            std = _standardize_pipeline(raw, all_symbols, mc_aligned, industry_map)
            std_panels.append(std)

        # --- Step 5: Equal-weight nanmean (min MIN_VALID_SUBFACTORS) ---
        stacked = np.stack(std_panels, axis=0)  # (5, N, T)
        valid_count = np.sum(~np.isnan(stacked), axis=0)   # (N, T)
        composite = np.nanmean(stacked, axis=0)             # (N, T)
        composite[valid_count < MIN_VALID_SUBFACTORS] = np.nan

        # --- Step 6: Post-process composite ---
        composite = _standardize_pipeline(composite, all_symbols, mc_aligned, industry_map)
        composite = composite.astype(np.float64)

        nan_ratio = np.isnan(composite).mean()
        if nan_ratio > 0.8:
            warnings.warn(
                f"{FACTOR_NAME} NaN ratio is high: {nan_ratio:.1%}, please check sub-factors"
            )

        return FactorData(
            values=composite,
            symbols=all_symbols,
            dates=ref_dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python rd_composite.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("RD_COMPOSITE factor smoke test")
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

    print(f"\n[Step 2] Data validation: check sub-factor fields")
    v, s, d = fd.get_daily_panel("q_ps_rade_c")
    print(f"  q_ps_rade_c (R&D): shape={v.shape}, NaN={np.isnan(v).mean():.1%}")
    assert v.size > 0, "q_ps_rade_c returned empty"

    v2, s2, d2 = fd.get_daily_panel("q_ps_se_c")
    print(f"  q_ps_se_c (SE): shape={v2.shape}, NaN={np.isnan(v2).mean():.1%}")
    assert v2.size > 0, "q_ps_se_c returned empty"

    mc_v, mc_s, mc_d = fd.get_market_cap_panel()
    print(f"  mc panel: shape={mc_v.shape}, NaN={np.isnan(mc_v).mean():.1%}")
    assert mc_v.size > 0, "mc panel returned empty"

    print(f"\n[Step 3] Compute RD_COMPOSITE factor")
    calculator = RD_COMPOSITE()
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
