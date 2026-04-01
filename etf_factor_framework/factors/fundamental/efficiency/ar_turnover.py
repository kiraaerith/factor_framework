"""
应收账款周转率因子（AR Turnover）

数据字段：lixinger financial_statements.q_m_ar_tor_t
因子方向：正向（应收账款周转率越高越好 - 表示收款效率高）
因子类别：运营效率

q_m_ar_tor_t 说明：
- q: 季度粒度
- m: 衍生指标（metrics）
- ar_tor: 应收账款周转率（应收账款周转次数）
- t: 时间加权（TTM / 滚动12个月）

应收账款周转率 = 营收 / 平均应收账款
- 值越高表示公司收款效率越高
- 值越低可能表示：坏账风险、销售下降、赊销增加等
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

FIELD = "q_m_ar_tor_t"
FACTOR_NAME = "AR_TURNOVER"
FACTOR_DIRECTION = 1  # 正向


class AR_TURNOVER(FundamentalFactorCalculator):
    """
    应收账款周转率因子

    使用 lixinger 的 q_m_ar_tor_t（应收账款周转率 TTM），
    以 report_date 为信号生效日，月度调仓。
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
        计算应收账款周转率因子日频面板

        Returns:
            FactorData: N股 × T日，值为 q_m_ar_tor_t（周转次数，如 2.5 表示周转2.5次）
        """
        values, symbols, dates = fundamental_data.get_daily_panel(FIELD)

        if values.size == 0:
            raise ValueError(f"AR_TURNOVER 因子：get_daily_panel('{FIELD}') 返回空数组，"
                             "请检查 lixinger 数据库中是否有该字段数据")

        nan_ratio = np.isnan(values).sum() / values.size
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(f"AR_TURNOVER 因子 NaN 比例偏高：{nan_ratio:.1%}，请检查数据")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# 测试（python ar_turnover.py 直接运行）
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("应收账款周转率因子测试")
    print("=" * 60)

    # 用较短时间段和少量股票快速验证
    TEST_START = "2022-01-01"
    TEST_END   = "2023-12-31"
    # 取几只知名股票做验证
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

    print(f"\n[Step 1] 加载 FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print(f"\n[Step 2] 计算应收账款周转率因子")
    calculator = AR_TURNOVER()
    factor = calculator.calculate(fd)

    print(f"\n因子 shape: {factor.shape}")
    print(f"symbols: {factor.symbols}")
    print(f"日期范围: {pd.Timestamp(factor.dates[0]).date()} ~ {pd.Timestamp(factor.dates[-1]).date()}")

    nan_ratio = np.isnan(factor.values).sum() / factor.values.size
    print(f"NaN 比例: {nan_ratio:.1%}")

    print(f"\n因子值样本（前5行，前5列）：")
    sample_df = pd.DataFrame(factor.values[:5, :5], index=factor.symbols[:5], columns=factor.dates[:5])
    print(sample_df.to_string())
