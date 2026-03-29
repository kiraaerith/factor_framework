"""
ROIC 因子（投入资本回报率）

数据字段：lixinger financial_statements.q_m_roic_t
因子方向：正向（ROIC 越高越好）
因子类别：盈利能力

q_m_roic_t 说明：
- q: 季度粒度
- m: 衍生指标（metrics）
- roic: 投入资本回报率
- t: 时间加权（TTM / 滚动12个月）
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

FIELD = "q_m_roic_t"
FACTOR_NAME = "ROIC"
FACTOR_DIRECTION = 1  # 正向


class ROIC(FundamentalFactorCalculator):
    """
    投入资本回报率（ROIC）因子

    使用 lixinger 的 q_m_roic_t（季度 ROIC，时间加权），
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
        计算 ROIC 因子日频面板

        Returns:
            FactorData: N股 × T日，值为 q_m_roic_t（百分比，如 0.12 = 12%）
        """
        values, symbols, dates = fundamental_data.get_daily_panel(FIELD)

        if values.size == 0:
            raise ValueError(f"ROIC 因子：get_daily_panel('{FIELD}') 返回空数组，"
                             "请检查 lixinger 数据库中是否有该字段数据")

        nan_ratio = np.isnan(values).sum() / values.size
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(f"ROIC 因子 NaN 比例偏高：{nan_ratio:.1%}，请检查数据")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            factor_type=self.factor_type,
            params=self.params,
        )


# ------------------------------------------------------------------
# 测试（python roic.py 直接运行）
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("ROIC 因子测试")
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

    print(f"\n[Step 2] 计算 ROIC 因子")
    calculator = ROIC()
    factor = calculator.calculate(fd)

    print(f"\n因子 shape: {factor.shape}")
    print(f"symbols: {factor.symbols}")
    print(f"日期范围: {pd.Timestamp(factor.dates[0]).date()} ~ {pd.Timestamp(factor.dates[-1]).date()}")

    nan_ratio = np.isnan(factor.values).sum() / factor.values.size
    print(f"NaN 比例: {nan_ratio:.1%}")

    print(f"\n因子值样本（前5行，前5列）：")
    sample_df = pd.DataFrame(factor.values[:5, :5], index=factor.symbols[:5], columns=factor.dates[:5])
    print(sample_df.to_string())

    print(f"\n因子值统计（最后一个截面）：")
    last_cs = factor.values[:, -1]
    last_cs = last_cs[~np.isnan(last_cs)]
    print(f"  均值: {last_cs.mean():.4f}")
    print(f"  标准差: {last_cs.std():.4f}")
    print(f"  最小值: {last_cs.min():.4f}")
    print(f"  最大值: {last_cs.max():.4f}")

    print(f"\n[Step 3] 泄露检测")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    detector = FundamentalLeakageDetector(split_ratio=0.7)
    report = detector.detect(calculator, fd)
    report.print_report()

    if report.has_leakage:
        print("\n[FAIL] 检测到数据泄露！请检查因子计算逻辑。")
    else:
        print("\n[PASS] 无数据泄露，因子代码正确。")
