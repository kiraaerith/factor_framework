"""
INTEREST_COVERAGE 因子（利息覆盖率）

衡量公司用营业利润支付利息费用的能力，反映财务是否可持续。

Formula:
    INTEREST_COVERAGE = EBIT / 财务费用
                      = q_ps_ebit_c / q_ps_fe_c

Data fields from lixinger.financial_statements:
    - q_ps_ebit_c : 息税前利润(EBIT, 单季)，非空率 100.0%
    - q_ps_fe_c   : 财务费用(单季)，非空率 ~95%

Factor direction: +1
    利息覆盖率越高，说明企业盈利能力越强，偿还利息的能力越强，财务风险越低。
    不能覆盖利息费用（比值 < 1）或覆盖率很低 → 财务压力大 → 预期超额收益越低

    特殊边界：
    - fe_c <= 0      → 无利息费用（金融费用为负或零），设为 NaN
                       （无债务或财务费用为负的公司不适用此因子）
    - ebit < 0       →存在亏损，保留负值（极端风险信号）
    - 无法计算       → NaN

因子类别: 质量 - 财务安全边际 (Quality / Safety)

市值过滤：mc < MC_MIN_BILLION 亿元时当日因子值置 NaN，与框架其他质量因子保持一致。

References:
    - Interest Coverage Ratio: 标准财务杠杆评估指标
    - Damodaran (2012): "Investment Valuation" — 债务可持续性分析
    - A股高杠杆去杠杆周期中，利息覆盖率回落通常伴随风险预期上升
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

FACTOR_NAME = "INTEREST_COVERAGE"
FACTOR_DIRECTION = 1
MC_MIN_BILLION = 50.0  # 市值下限（亿元）

_FIELDS = {
    "ebit": "q_ps_ebit_c",  # 息税前利润（单季），非空率 100.0%
    "fe":   "q_ps_fe_c",    # 财务费用（单季），非空率 ~95%
}


class InterestCoverage(FundamentalFactorCalculator):
    """
    利息覆盖率因子：EBIT / 财务费用

    参数
    ----
    无额外参数。因子由两个利润表项目直接计算，无需窗口/滞后设置。

    说明
    ----
    - 无利息负担的公司（fe_c ≤ 0）因子值置 NaN，不参与排序
    - 市值较小的公司（< 50亿）因子值置 NaN，避免微市值公司噪音
    - 仓位映射层会对极端值进行缩尾处理（winsorization）
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
    # Main calculation
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData, pricevol_data=None) -> FactorData:
        """
        计算利息覆盖率日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为 float（可以是负数）或 NaN。
                     NaN 出现在：
                     - 数据缺失的场景
                     - 财务费用 ≤ 0（无利息负担）
                     - 市值 < 50 亿元
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        T = len(trading_dates)

        # ------------------------------------------------------------------
        # Step 1: Load raw panels
        # ------------------------------------------------------------------
        all_symbols: set = set()
        raw_panels: dict = {}

        for key, field in _FIELDS.items():
            try:
                vals, syms, dts = fundamental_data.get_daily_panel(field)
                df = pd.DataFrame(
                    vals,
                    index=syms.tolist(),
                    columns=pd.DatetimeIndex(dts),
                )
                raw_panels[key] = df
                all_symbols.update(syms.tolist())
            except Exception as exc:
                warnings.warn(
                    f"InterestCoverage: failed to load field '{field}' "
                    f"(key={key}): {exc}. Factor will be all NaN."
                )
                raw_panels[key] = pd.DataFrame(dtype=np.float64)

        if not all_symbols:
            empty = np.empty((0, T), dtype=np.float64)
            return FactorData(
                values=empty,
                symbols=np.array([], dtype=str),
                dates=np.array(trading_dates, dtype="datetime64[ns]"),
                name=self.name,
                factor_type=self.factor_type,
                params=self.params,
            )

        all_symbols_sorted = sorted(all_symbols)
        N = len(all_symbols_sorted)
        td_idx = pd.DatetimeIndex(trading_dates)

        # ------------------------------------------------------------------
        # Step 2: Align to (all_symbols, trading_dates)
        # ------------------------------------------------------------------
        def _align(df: pd.DataFrame) -> np.ndarray:
            if df.empty:
                return np.full((N, T), np.nan, dtype=np.float64)
            aligned = df.reindex(index=all_symbols_sorted, columns=td_idx)
            return aligned.values.astype(np.float64)

        ebit_arr = _align(raw_panels["ebit"])  # (N, T)
        fe_arr   = _align(raw_panels["fe"])    # (N, T)

        # ------------------------------------------------------------------
        # Step 3: Compute ratio = ebit / fe
        #   - fe <= 0           → NaN  (无利息负担，因子无意义)
        #   - ebit无效          → NaN
        #   - ebit < 0, fe > 0  → 负数 (亏损且有利息费用，极端风险)
        #   - ebit > 0, fe > 0  → 正数 (正常情况)
        # ------------------------------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                ~np.isnan(ebit_arr) & ~np.isnan(fe_arr) & (fe_arr > 0),
                ebit_arr / fe_arr,
                np.nan,
            )

        # ------------------------------------------------------------------
        # Step 4: Market-cap floor filter
        # ------------------------------------------------------------------
        mc_vals, mc_syms, _ = fundamental_data.get_market_cap_panel()
        mc_sym_idx = {s: i for i, s in enumerate(mc_syms.tolist())}
        mc_aligned = np.full((N, T), np.nan, dtype=np.float64)
        for i, sym in enumerate(all_symbols_sorted):
            j = mc_sym_idx.get(sym)
            if j is not None:
                mc_aligned[i, :] = mc_vals[j, :]

        invalid_mc = np.isnan(mc_aligned) | (mc_aligned < MC_MIN_BILLION)
        ratio[invalid_mc] = np.nan

        n_mc_masked = int(invalid_mc.sum())
        if n_mc_masked > 0:
            n_no_mc = int(np.isnan(mc_aligned).sum())
            n_below = n_mc_masked - n_no_mc
            print(
                f"  - 市值过滤: 无市值数据 {n_no_mc} 个股-日, "
                f"<{MC_MIN_BILLION:.0f}亿 {n_below} 个股-日"
            )

        nan_ratio = np.isnan(ratio).mean() if ratio.size > 0 else 1.0
        if nan_ratio > 0.9:
            warnings.warn(
                f"InterestCoverage NaN ratio is very high ({nan_ratio:.1%}). "
                "Check if required fields (q_ps_ebit_c, q_ps_fe_c) exist in the lixinger database."
            )

        return FactorData(
            values=ratio,
            symbols=np.array(all_symbols_sorted),
            dates=np.array(trading_dates, dtype="datetime64[ns]"),
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# Smoke test (python interest_coverage.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("InterestCoverage factor smoke test")
    print("=" * 60)

    TEST_START = "2019-01-01"
    TEST_END   = "2024-12-31"

    try:
        fd = FundamentalData(start_date=TEST_START, end_date=TEST_END)
        calc = InterestCoverage()
        factor = calc.calculate(fd)

        print(f"\nFactor name:    {factor.name}")
        print(f"Factor type:    {factor.factor_type}")
        print(f"Factor params:  {factor.params}")
        print(f"Shape:          {factor.shape}")
        print(f"Symbols:        {len(factor.symbols)} unique")
        print(f"Date range:     {factor.dates[0]} to {factor.dates[-1]}")
        print(f"NaN ratio:      {np.isnan(factor.values).mean():.1%}")

        # Quick sanity check
        valid_mask = ~np.isnan(factor.values)
        if valid_mask.any():
            valid_vals = factor.values[valid_mask]
            print(f"Value range:    [{valid_vals.min():.3f}, {valid_vals.max():.3f}]")
            print(f"Mean value:     {valid_vals.mean():.3f}")
            print(f"Median value:   {np.median(valid_vals):.3f}")

        print("\n✓ Smoke test passed!")

    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
