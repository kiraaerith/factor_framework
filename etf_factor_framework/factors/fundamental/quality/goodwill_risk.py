"""
Goodwill Risk Factor（商誉风险因子）

计算逻辑：
    GOODWILL_RISK = 商誉 / 净资产 = q_bs_gw_t / q_bs_toe_t

经济含义：
    商誉是企业并购溢价的沉淀资产，大额商誉一旦减值会直接冲击净利润和净资产。
    A股商誉暴雷事件频发，高商誉/净资产比率代表更大的减值风险和盈利不确定性。

边界处理：
    - toe <= 0（负净资产）→ NaN，负净资产公司本身已是极端情况，无法可靠排序
    - gw == 0            → ratio = 0，无商誉是正常情况，代表最低风险（保留）
    - gw < 0             → NaN，商誉不应为负，视为数据异常

因子方向：-1（比值越高 = 商誉风险越大 = 预期超额收益越低）
因子类别：质量 - 财务安全边际 (Quality / Safety)

市值过滤：mc < MC_MIN_BILLION 亿元时当日因子值置 NaN，与框架其他质量因子保持一致。
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

FACTOR_NAME = "GOODWILL_RISK"
FACTOR_DIRECTION = -1
MC_MIN_BILLION = 50.0  # 市值下限（亿元）

_FIELDS = {
    "gw":  "q_bs_gw_t",   # 商誉，非空率 100.0%
    "toe": "q_bs_toe_t",  # 所有者权益合计（净资产），非空率 93.7%
}


class GoodwillRisk(FundamentalFactorCalculator):
    """
    商誉风险因子：商誉 / 净资产

    参数
    ----
    无额外参数。因子由两个资产负债表余额字段直接计算，无需窗口/滞后设置。
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
        计算商誉风险日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为 ≥0 的 float 或 NaN。
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
                    f"GoodwillRisk: failed to load field '{field}' "
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

        gw_arr  = _align(raw_panels["gw"])   # (N, T)
        toe_arr = _align(raw_panels["toe"])  # (N, T)

        # ------------------------------------------------------------------
        # Step 3: Compute ratio = gw / toe
        #   - toe <= 0          → NaN  (负净资产无意义)
        #   - gw < 0            → NaN  (商誉不应为负，数据异常)
        #   - gw == 0, toe > 0  → 0.0  (合法：无商誉)
        # ------------------------------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                ~np.isnan(gw_arr) & ~np.isnan(toe_arr)
                & (toe_arr > 0) & (gw_arr >= 0),
                gw_arr / toe_arr,
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
                f"GoodwillRisk NaN ratio is very high ({nan_ratio:.1%}). "
                "Check if required fields exist in the lixinger database."
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
# Smoke test (python goodwill_risk.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("GoodwillRisk factor smoke test")
    print("=" * 60)

    TEST_START = "2019-01-01"
    TEST_END   = "2024-12-31"

    print(f"\n[Step 1] Load FundamentalData ({TEST_START} ~ {TEST_END}, full market)")
    fd = FundamentalData(start_date=TEST_START, end_date=TEST_END)

    print("\n[Step 2] Compute GoodwillRisk factor")
    calculator = GoodwillRisk()
    result = calculator.calculate(fd)

    print(f"\nFactor shape     : {result.values.shape}")
    print(f"Symbols (first 5): {result.symbols[:5].tolist()}")
    print(f"Date range       : {pd.Timestamp(result.dates[0]).date()} ~ "
          f"{pd.Timestamp(result.dates[-1]).date()}")
    print(f"Direction        : {result.params['direction']}")

    assert result.values.ndim == 2, "values must be 2-D"
    assert result.values.dtype == np.float64, f"expected float64, got {result.values.dtype}"

    nan_ratio = np.isnan(result.values).mean()
    print(f"NaN ratio        : {nan_ratio:.1%}")
    assert nan_ratio < 0.95, f"NaN ratio too high: {nan_ratio:.1%}"

    valid = result.values[~np.isnan(result.values)]
    print(f"Value range      : [{valid.min():.4f}, {valid.max():.4f}]")
    print(f"Median           : {np.median(valid):.4f}")

    # 商誉为 0 的股票应占多数（大量非并购型公司），检查 0 的比例
    zero_count = int((valid == 0.0).sum())
    print(f"Zero-goodwill    : {zero_count} ({zero_count / len(valid):.1%} of valid values)")

    print("\n[PASS] Smoke test completed successfully.")
