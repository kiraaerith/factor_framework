"""
PB 因子（市净率）

数据字段：lixinger fundamental.pb
因子方向：反向（低 PB 股票预期收益更好）
因子类别：估值 - 市净率

pb 说明：
- 来源：lixinger fundamental 表（日频估值，每日更新）
- 含义：市净率 = 总市值 / 净资产
- 无季报延迟：T 日 pb 在 T 日收盘前已知，无未来泄露风险
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

FIELD = "pb"
FACTOR_NAME = "PB"
FACTOR_DIRECTION = -1  # 反向：低 PB 更好


class PB(FundamentalFactorCalculator):
    """
    市净率（Price-to-Book）因子

    使用 lixinger 的 fundamental.pb（日频市净率）。
    PB 是日频市场数据，无季报延迟，T 日数据 T 日已知，无未来泄露。
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
        计算 PB 因子日频面板

        Returns:
            FactorData: N股 × T日，值为 pb（市净率）
        """
        values, symbols, dates = fundamental_data.get_valuation_panel(FIELD)

        if values.size == 0:
            raise ValueError(f"PB 因子：get_valuation_panel('{FIELD}') 返回空数组，"
                             "请检查 lixinger 数据库中是否有该字段数据")

        nan_ratio = np.isnan(values).sum() / values.size
        if nan_ratio > 0.8:
            import warnings
            warnings.warn(f"PB 因子 NaN 比例偏高：{nan_ratio:.1%}，请检查数据")

        return FactorData(
            values=values,
            symbols=symbols,
            dates=dates,
            name=self.name,
            params=self.params,
        )


# ------------------------------------------------------------------
# 测试（python pb.py 直接运行）
# ------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("PB 因子 smoke test")
    print("=" * 60)

    TEST_START = "2022-01-01"
    TEST_END   = "2023-12-31"
    TEST_CODES = ["600519", "000858", "601318", "000333", "600036"]

    print(f"\n[Step 1] 加载 FundamentalData ({TEST_START} ~ {TEST_END})")
    fd = FundamentalData(
        start_date=TEST_START,
        end_date=TEST_END,
        stock_codes=TEST_CODES,
    )

    print(f"\n[Step 2] 计算 PB 因子")
    calculator = PB()
    factor = calculator.calculate(fd)

    print(f"\n因子 shape: {factor.shape}")
    print(f"symbols 数量: {len(factor.symbols)}")
    print(f"日期范围: {pd.Timestamp(factor.dates[0]).date()} ~ {pd.Timestamp(factor.dates[-1]).date()}")

    nan_ratio = np.isnan(factor.values).sum() / factor.values.size
    print(f"NaN 比例: {nan_ratio:.1%}")

    # 打印 5 只代表股的 PB 均值
    print(f"\n代表股 PB 均值（2022~2023）：")
    target_symbols = [f"SHSE.{c}" if c.startswith('6') else f"SZSE.{c}" for c in TEST_CODES]
    for sym in target_symbols:
        idx = np.where(factor.symbols == sym)[0]
        if len(idx):
            row = factor.values[idx[0]]
            valid = row[~np.isnan(row)]
            if len(valid):
                print(f"  {sym}: mean_pb={valid.mean():.2f}, min={valid.min():.2f}, max={valid.max():.2f}")

    print(f"\n因子值统计（最后一个截面）：")
    last_cs = factor.values[:, -1]
    last_cs = last_cs[~np.isnan(last_cs)]
    print(f"  股票数: {len(last_cs)}")
    print(f"  均值: {last_cs.mean():.2f}")
    print(f"  中位数: {np.median(last_cs):.2f}")
    print(f"  最小值: {last_cs.min():.2f}")
    print(f"  最大值: {last_cs.max():.2f}")

    print(f"\n[Step 3] 泄露检测")
    from factors.fundamental.fundamental_leakage_detector import FundamentalLeakageDetector

    detector = FundamentalLeakageDetector(split_ratio=0.7)
    report = detector.detect(calculator, fd)
    report.print_report()

    if report.has_leakage:
        print("\n[FAIL] 检测到数据泄露！请检查因子计算逻辑。")
    else:
        print("\n[PASS] 无数据泄露，因子代码正确。")
