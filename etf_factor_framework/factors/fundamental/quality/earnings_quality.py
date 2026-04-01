"""
EARNINGS_QUALITY 因子（盈余质量：经营现金流/净利润比率）

衡量公司净利润中现金含量的比率，反映盈余质量的核心维度之一。
OCF/NI 比率越高，说明利润的现金含量越高、会计操纵空间越小，盈余质量越好。

Formula:
    EARNINGS_QUALITY = OCF / Net Income (TTM)
                     = q_m_ncffoa_np_r_t

Data fields from lixinger.financial_statements:
    - q_m_ncffoa_np_r_t : 经营现金流对净利润比(TTM)
                          由 lixinger 预计算，直接使用，非空率 62.1%

Factor direction: +1 (higher OCF/NI ratio → better earnings quality → more desirable)
Factor category: quality

Notes:
    - Uses report_date for forward-fill (no future data leakage).
    - Extreme values (e.g. NI ≈ 0) can produce arbitrarily large ratios;
      outliers are handled by grid-level winsorization during evaluation,
      not clipped here to preserve information.
    - Market-cap floor: stocks with mc < 50亿 (or unknown mc) are set to NaN
      to avoid micro-cap noise dominating the factor.  Same threshold and
      alignment logic as piotroski_fscore.py.

References:
    - Sloan (1996): "Do Stock Prices Fully Reflect Information in Accruals
      and Cash Flows About Future Earnings?" The Accounting Review.
    - Dechow et al. (1998): "The Relation Between Earnings and Cash Flows.
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

FIELD = "q_m_ncffoa_np_r_t"   # OCF/NI ratio TTM, from lixinger financial_statements
FACTOR_NAME = "EARNINGS_QUALITY"
FACTOR_DIRECTION = 1           # +1: higher OCF/NI → better

MC_MIN_BILLION = 50.0          # 市值下限（亿元），与 piotroski_fscore.py 保持一致


class EarningsQuality(FundamentalFactorCalculator):
    """
    盈余质量因子 (Earnings Quality)

    使用 lixinger 的 q_m_ncffoa_np_r_t（经营现金流/净利润 TTM 比率），
    以 report_date 为信号生效日，避免未来数据泄露。

    市值过滤：日市值 < 50亿 或无市值数据的股票当日因子值置 NaN，
    防止超小市值公司的极端值干扰横截面排名。
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

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算 EARNINGS_QUALITY 因子日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为 q_m_ncffoa_np_r_t（TTM OCF/NI 比率），
                     市值不足 MC_MIN_BILLION 亿元的股票当日置 NaN。
        """
        # ------------------------------------------------------------------
        # Step 1: Load OCF/NI ratio panel
        # ------------------------------------------------------------------
        try:
            values, symbols, dates = fundamental_data.get_daily_panel(FIELD)
        except Exception as exc:
            raise ValueError(
                f"EarningsQuality: failed to load field '{FIELD}': {exc}"
            ) from exc

        if values.size == 0:
            raise ValueError(
                f"EarningsQuality: get_daily_panel('{FIELD}') returned empty array. "
                "Please check lixinger database has this field."
            )

        values = values.copy().astype(np.float64)
        N, T = values.shape
        symbols_list = symbols.tolist()

        # ------------------------------------------------------------------
        # Step 2: Apply market-cap floor
        #   Replicate pattern from piotroski_fscore.py:
        #   align mc panel to factor symbols, then mask invalid entries.
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
            values[invalid_mc] = np.nan

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
                f"EarningsQuality: market-cap filter skipped due to error: {exc}. "
                "Factor values will NOT be market-cap filtered."
            )

        # ------------------------------------------------------------------
        # Step 3: Quality check
        # ------------------------------------------------------------------
        nan_ratio = np.isnan(values).mean() if values.size > 0 else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                f"EarningsQuality NaN ratio is very high ({nan_ratio:.1%}). "
                "Check if field 'q_m_ncffoa_np_r_t' is populated in the lixinger database."
            )

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (run: python earnings_quality.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("EARNINGS_QUALITY factor smoke test")
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

    print("\n[Step 2] Calculate EARNINGS_QUALITY factor")
    calculator = EarningsQuality()
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
