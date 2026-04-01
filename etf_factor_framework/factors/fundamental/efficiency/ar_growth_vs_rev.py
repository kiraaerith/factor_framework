"""
AR_GROWTH_VS_REV 因子（应收账款增速 - 营收增速）

衡量应收账款增速相对营收增速的偏差。
当 AR 增速显著高于营收增速时，可能预示坏账风险累积或虚增收入。

Formula:
    AR_GROWTH_VS_REV = AR_YoY - Rev_YoY

    其中：
      AR_YoY  = (q_bs_ar_t - q_bs_ar_t.shift(lag)) / abs(q_bs_ar_t.shift(lag))
      Rev_YoY = (q_ps_toi_c - q_ps_toi_c.shift(lag)) / abs(q_ps_toi_c.shift(lag))

    通过在日频面板上向前滚动 lag 个交易日（默认 ~250 交易日 ≈ 1 年）
    近似得到同比增长率。两个增长率采用一致的计算方法以保证可比性。

Data fields from lixinger.financial_statements:
    - q_bs_ar_t  : 应收账款（期末），非空率 99.2%
    - q_ps_toi_c : 营业总收入（单季），非空率 59.6%

Factor direction: -1
    AR 增速越大于营收增速，说明收款质量下降，预期超额收益越低。

边界处理：
    - lag 期前数值为 NaN 或 0 → 当期 YoY 置 NaN
    - 极端 YoY 值（绝对值 > extreme_clip）截断，防止极端季报扰动
    - 两个 YoY 任意一个为 NaN → 结果置 NaN

参数：
    lag        : 向后取参照期的交易日数，默认 250（≈ 1 年）
    extreme_clip : YoY 截断阈值（绝对值），默认 10.0（1000%），用于过滤异常季报

因子类别: 运营效率 - 收款质量 (Efficiency / Receivables Quality)

References:
    - Beneish (1999): 应收账款增速 vs 收入增速差异是 M-Score 核心指标之一
    - A股实证：信用销售高增长时期 AR 超速成长是盈余操控的预警信号
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

FACTOR_NAME = "AR_GROWTH_VS_REV"
FACTOR_DIRECTION = -1  # 负向：AR 增速 > Rev 增速越明显，预期表现越差

_DEFAULT_LAG = 250          # 默认回看交易日数（≈ 1 年同比）
_DEFAULT_EXTREME_CLIP = 10.0  # YoY 极端值截断阈值（绝对值，即 ±1000%）

_FIELDS = {
    "ar":  "q_bs_ar_t",   # 应收账款（期末），非空率 99.2%
    "rev": "q_ps_toi_c",  # 营业总收入（单季），非空率 59.6%
}


def _yoy_growth(arr: np.ndarray, lag: int, extreme_clip: float) -> np.ndarray:
    """
    逐时间步计算 YoY 增长率（在日频面板上滚动）。

    arr        : shape (N, T)，已 forward-fill 的日频面板
    lag        : 回看时间步数（交易日）
    extreme_clip : 截断阈值（绝对值）

    Returns: shape (N, T)，YoY 增长率；无法计算的位置为 NaN
    """
    T = arr.shape[1]
    if T <= lag:
        return np.full_like(arr, np.nan, dtype=np.float64)

    prev = np.full_like(arr, np.nan, dtype=np.float64)
    prev[:, lag:] = arr[:, :T - lag]  # lag 步前的值

    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(
            ~np.isnan(prev) & ~np.isnan(arr) & (np.abs(prev) > 0),
            (arr - prev) / np.abs(prev),
            np.nan,
        )

    # 截断极端值
    growth = np.where(
        np.abs(growth) > extreme_clip,
        np.nan,
        growth,
    )
    return growth


class ARGrowthVsRev(FundamentalFactorCalculator):
    """
    应收账款增速 vs 营收增速因子（AR_GROWTH_VS_REV）

    计算 AR 同比增长率 - 营收同比增长率，捕捉 AR 增速异常偏高的风险信号。

    参数
    ----
    lag : int
        回看交易日数，默认 250（≈ 1 年同比）。
    extreme_clip : float
        YoY 增长率极端值截断阈值（绝对值），默认 10.0（±1000%）。
    """

    def __init__(
        self,
        lag: int = _DEFAULT_LAG,
        extreme_clip: float = _DEFAULT_EXTREME_CLIP,
    ):
        self._lag = lag
        self._extreme_clip = extreme_clip

    @property
    def name(self) -> str:
        return f"{FACTOR_NAME}_lag{self._lag}"

    @property
    def factor_type(self) -> str:
        return FACTOR_NAME

    @property
    def params(self) -> dict:
        return {
            "direction": FACTOR_DIRECTION,
            "lag": self._lag,
            "extreme_clip": self._extreme_clip,
        }

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def calculate(self, fundamental_data: FundamentalData) -> FactorData:
        """
        计算 AR_GROWTH_VS_REV 日频面板。

        Returns
        -------
        FactorData : shape (N, T)，值为 AR_YoY - Rev_YoY。
                     NaN 出现在：
                     - 历史数据不足 lag 期
                     - AR 或 Rev 数据缺失
                     - 参照期分母为 0 或 NaN
                     - YoY 绝对值超过 extreme_clip（截断）
        """
        fundamental_data._load_raw_data()
        trading_dates = fundamental_data._get_trading_dates()
        T = len(trading_dates)

        # ------------------------------------------------------------------
        # Step 1: Load raw panels for AR and Rev
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
                    f"ARGrowthVsRev: failed to load field '{field}' "
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

        ar_arr  = _align(raw_panels["ar"])   # (N, T) 应收账款
        rev_arr = _align(raw_panels["rev"])  # (N, T) 营业总收入（单季）

        # ------------------------------------------------------------------
        # Step 3: Compute YoY for AR and Rev using rolling lag-day shift
        # ------------------------------------------------------------------
        ar_yoy  = _yoy_growth(ar_arr,  self._lag, self._extreme_clip)
        rev_yoy = _yoy_growth(rev_arr, self._lag, self._extreme_clip)

        # ------------------------------------------------------------------
        # Step 4: Factor = AR_YoY - Rev_YoY
        # ------------------------------------------------------------------
        factor_arr = np.where(
            ~np.isnan(ar_yoy) & ~np.isnan(rev_yoy),
            ar_yoy - rev_yoy,
            np.nan,
        )

        # ------------------------------------------------------------------
        # Step 5: Inf cleanup
        # ------------------------------------------------------------------
        factor_arr = np.where(np.isinf(factor_arr), np.nan, factor_arr)
        factor_arr = factor_arr.astype(np.float64)

        nan_ratio = np.isnan(factor_arr).mean() if factor_arr.size > 0 else 1.0
        if nan_ratio > 0.9:
            warnings.warn(
                f"AR_GROWTH_VS_REV NaN 比例偏高: {nan_ratio:.1%}，"
                "请检查 q_bs_ar_t / q_ps_toi_c 字段数据"
            )

        return FactorData(
            values=factor_arr,
            symbols=np.array(all_symbols_sorted),
            dates=np.array(trading_dates, dtype="datetime64[ns]"),
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# 本地测试（python ar_growth_vs_rev.py 直接运行）
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("AR_GROWTH_VS_REV 因子测试")
    print("=" * 60)

    TEST_START = "2022-01-01"
    TEST_END   = "2023-12-31"
    TEST_CODES = ["600000", "000001", "600036", "000651", "601318"]

    from factors.fundamental.fundamental_data import FundamentalData

    fd = FundamentalData(TEST_START, TEST_END, stock_codes=TEST_CODES)
    calc = ARGrowthVsRev(lag=250)
    result = calc.calculate(fd)

    print(f"\n因子名称 : {result.name}")
    print(f"面板形状 : {result.values.shape}")
    print(f"股票数量 : {len(result.symbols)}")
    print(f"日期范围 : {result.dates[0]} ~ {result.dates[-1]}")
    nan_pct = np.isnan(result.values).mean()
    print(f"NaN 占比  : {nan_pct:.1%}")

    # 展示部分样本值
    df_preview = pd.DataFrame(
        result.values,
        index=result.symbols,
        columns=pd.DatetimeIndex(result.dates),
    )
    print("\n最近 5 个交易日因子值（AR_YoY - Rev_YoY）：")
    print(df_preview.iloc[:, -5:].round(4))
