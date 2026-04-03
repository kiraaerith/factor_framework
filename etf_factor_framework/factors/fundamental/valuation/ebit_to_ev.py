"""
EBIT_TO_EV 因子（企业价值收益率：EBIT/EV）

EV/EBIT 剔除了资本结构差异，是比 P/E 更纯粹的经营盈利估值指标。
本因子取其倒数 EBIT/EV，使方向为正向（值越大 = 企业越便宜 = 越好）。

Formula:
    EBIT_TO_EV = 1 / ev_ebit_r

Data fields from lixinger.fundamental (daily valuation):
    - ev_ebit_r : EV/EBIT，由 lixinger 日频计算，非空率 83.1%

Factor direction: +1 (higher EBIT/EV → cheaper enterprise → more desirable)
Factor category: valuation

Notes:
    - ev_ebit_r comes from the daily fundamental table (no quarterly lag).
      T-day value is known as of T-day close → no future data leakage.
    - When ev_ebit_r ≤ 0 (negative EBIT) the ratio is economically
      meaningless as a value signal; those cells are set to NaN.
    - Inf values (ev_ebit_r ≈ 0) are also set to NaN.
    - Market-cap floor: stocks with mc < 50亿 (or unknown mc) are set to NaN
      to filter out micro-cap noise.  Same threshold and alignment logic as
      piotroski_fscore.py.
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

FIELD = "ev_ebit_r"         # EV/EBIT ratio, from lixinger.fundamental (daily)
FACTOR_NAME = "EBIT_TO_EV"
FACTOR_DIRECTION = 1        # +1: higher EBIT/EV → cheaper → better

MC_MIN_BILLION = 50.0       # 市值下限（亿元），与 piotroski_fscore.py 保持一致


class EbitToEv(FundamentalFactorCalculator):
    """
    EBIT/EV 因子（企业价值收益率）

    使用 lixinger fundamental.ev_ebit_r（日频 EV/EBIT），取倒数得到
    EBIT/EV。负值（EBIT < 0）和 inf 值置 NaN。
    市值过滤：日市值 < 50亿 或无市值数据的股票当日因子值置 NaN。
    """

    @property
    def name(self) -> str:
        return FACTOR_NAME

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {"field": FIELD, "direction": FACTOR_DIRECTION}

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        计算 EBIT_TO_EV 因子日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为 1 / ev_ebit_r；
                     ev_ebit_r ≤ 0 或市值 < MC_MIN_BILLION 的股票-日置 NaN。
        """
        # ------------------------------------------------------------------
        # Step 1: Load EV/EBIT panel from daily fundamental table
        # ------------------------------------------------------------------
        try:
            values, symbols, dates = fundamental_data.get_valuation_panel(FIELD)
        except Exception as exc:
            raise ValueError(
                f"EbitToEv: failed to load field '{FIELD}': {exc}"
            ) from exc

        if values.size == 0:
            raise ValueError(
                f"EbitToEv: get_valuation_panel('{FIELD}') returned empty array. "
                "Please check lixinger database has this field."
            )

        values = values.copy().astype(np.float64)
        N, T = values.shape
        symbols_list = symbols.tolist()

        # ------------------------------------------------------------------
        # Step 2: Compute inverse; mask non-positive ratios
        #   ev_ebit_r ≤ 0 means negative EBIT → not a valid value signal
        # ------------------------------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_values = np.where(
                ~np.isnan(values) & (values > 0),
                1.0 / values,
                np.nan,
            ).astype(np.float64)

        # Mask any remaining inf (ev_ebit_r extremely close to 0)
        inv_values = np.where(np.isinf(inv_values), np.nan, inv_values)

        # ------------------------------------------------------------------
        # Step 3: Apply market-cap floor
        # ------------------------------------------------------------------
        try:
            mc_vals, mc_syms, _ = fundamental_data.get_market_cap_panel()
            mc_sym_idx = {s: i for i, s in enumerate(mc_syms.tolist())}

            mc_aligned = np.full((N, T), np.nan, dtype=np.float64)
            for i, sym in enumerate(symbols_list):
                j = mc_sym_idx.get(sym)
                if j is not None:
                    mc_aligned[i, :] = mc_vals[j, :]

            invalid_mc = np.isnan(mc_aligned) | (mc_aligned < MC_MIN_BILLION)
            inv_values[invalid_mc] = np.nan

            n_mc_masked = int(invalid_mc.sum())
            if n_mc_masked > 0:
                n_no_mc = int(np.isnan(mc_aligned).sum())
                n_below = n_mc_masked - n_no_mc
                print(
                    f"  - 市值过滤: 无市值数据 {n_no_mc} 个股-日, "
                    f"<{MC_MIN_BILLION:.0f}亿 {n_below} 个股-日"
                )
        except Exception as exc:
            warnings.warn(
                f"EbitToEv: market-cap filter skipped due to error: {exc}. "
                "Factor values will NOT be market-cap filtered."
            )

        # ------------------------------------------------------------------
        # Step 4: Quality check
        # ------------------------------------------------------------------
        nan_ratio = np.isnan(inv_values).mean() if inv_values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"EBIT_TO_EV NaN ratio is very high ({nan_ratio:.1%}). "
                "Check if field 'ev_ebit_r' is populated in the lixinger database."
            )

        return FactorData(
            values=inv_values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python ebit_to_ev.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("EBIT_TO_EV factor smoke test")
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

    print("\n[Step 2] Calculate EBIT_TO_EV factor")
    calculator = EbitToEv()
    factor = calculator.calculate(fd)

    print(f"\nFactor shape: {factor.shape}")
    print(f"Symbols: {factor.symbols}")
    print(
        f"Date range: "
        f"{pd.Timestamp(factor.dates[0]).date()} ~ "
        f"{pd.Timestamp(factor.dates[-1]).date()}"
    )

    nan_ratio = np.isnan(factor.values).sum() / factor.values.size
    print(f"NaN ratio: {nan_ratio:.1%}")

    print("\nFactor values sample (first 5 rows, first 5 columns):")
    sample_df = pd.DataFrame(
        factor.values[:5, :5],
        index=factor.symbols[:5],
        columns=factor.dates[:5],
    )
    print(sample_df.to_string())

    print("\nFactor statistics (last cross-section):")
    last_cs = factor.values[:, -1]
    last_cs = last_cs[~np.isnan(last_cs)]
    if len(last_cs) > 0:
        print(f"  Count: {len(last_cs)}")
        print(f"  Mean:  {last_cs.mean():.4f}")
        print(f"  Std:   {last_cs.std():.4f}")
        print(f"  Min:   {last_cs.min():.4f}")
        print(f"  Max:   {last_cs.max():.4f}")
    else:
        print("  (no valid values in last cross-section)")

    print("\n[Step 3] Leakage detection")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    for split_ratio in [0.5, 0.6, 0.7, 0.8, 0.9]:
        detector = FundamentalLeakageDetector(split_ratio=split_ratio)
        report = detector.detect(calculator, fd)
        print(
            f"  split_ratio={split_ratio:.1f}: "
            f"leakage={report.get('has_leakage', 'N/A')}, "
            f"details={report.get('summary', report)}"
        )

    print("\nSmoke test completed.")
