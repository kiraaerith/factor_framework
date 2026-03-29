"""
放量缩量切分因子测试脚本
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.ohlcv_data import OHLCVData
from factors.ctc.volume_change_split import (
    HighVolChangeReturnSum,
    LowVolChangeReturnSum,
    HighVolChangeReturnStd,
    LowVolChangeReturnStd,
    HighVolChangeAmplitude,
    LowVolChangeAmplitude,
)


def create_test_data(n_assets: int = 5, n_periods: int = 60) -> OHLCVData:
    """创建测试数据"""
    symbols = [f'ETF{i}' for i in range(n_assets)]
    dates = pd.date_range('2024-01-01', periods=n_periods)
    
    # 生成随机价格数据
    np.random.seed(42)
    close = pd.DataFrame(
        np.random.randn(n_assets, n_periods).cumsum(axis=1) + 100,
        index=symbols, columns=dates
    )
    
    ohlcv = OHLCVData(
        open=close * 0.99,
        high=close * 1.02,
        low=close * 0.98,
        close=close,
        volume=pd.DataFrame(
            np.abs(np.random.randn(n_assets, n_periods)) * 1000,
            index=symbols, columns=dates
        )
    )
    return ohlcv


def test_factor_basic(FactorClass, factor_name):
    """测试因子基础功能"""
    print(f"\n{'='*60}")
    print(f"测试: {factor_name}")
    print(f"{'='*60}")
    
    # 准备数据
    ohlcv = create_test_data()
    print(f"数据形状: {ohlcv.shape}")
    
    # 创建因子
    factor = FactorClass(window=20, top_pct=0.2)
    print(f"因子名称: {factor.name}")
    print(f"因子类型: {factor.factor_type}")
    print(f"因子参数: {factor.params}")
    
    # 计算因子
    print("\n计算因子...")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    # 检查输出格式
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods), "因子形状与数据不匹配"
    assert list(factor_data.values.index) == list(ohlcv.symbols), "symbol索引不匹配"
    assert list(factor_data.values.columns) == list(ohlcv.dates), "日期列不匹配"
    
    print(f"\n[PASS] {factor_name} 基础功能测试通过")
    return True


def test_factor_with_real_data(FactorClass, factor_name):
    """使用真实数据测试因子"""
    print(f"\n{'='*60}")
    print(f"真实数据测试: {factor_name}")
    print(f"{'='*60}")
    
    # 加载真实数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    if not os.path.exists(csv_path):
        print(f"⚠️ 数据文件不存在: {csv_path}")
        print("跳过真实数据测试")
        return True
    
    df = pd.read_csv(csv_path)
    df['eob'] = pd.to_datetime(df['eob'])
    df.columns = [c.lower() for c in df.columns]
    
    ohlcv = OHLCVData.from_dataframe(
        df,
        symbol_col='symbol',
        date_col='eob',
        ohlcv_cols={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    )
    
    print(f"真实数据形状: {ohlcv.shape}")
    
    # 测试因子
    factor = FactorClass(window=20, top_pct=0.2)
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子计算完成，形状: {factor_data.shape}")
    print(f"因子值统计:")
    print(f"  - 均值: {factor_data.values.mean().mean():.4f}")
    print(f"  - 标准差: {factor_data.values.std().mean():.4f}")
    print(f"  - 最小值: {factor_data.values.min().min():.4f}")
    print(f"  - 最大值: {factor_data.values.max().max():.4f}")
    print(f"  - NaN比例: {factor_data.values.isna().sum().sum() / factor_data.values.size:.2%}")
    
    print(f"\n[PASS] {factor_name} 真实数据测试通过")
    return True


def main():
    """运行所有测试"""
    factors_to_test = [
        (HighVolChangeReturnSum, "HighVolChangeReturnSum"),
        (LowVolChangeReturnSum, "LowVolChangeReturnSum"),
        (HighVolChangeReturnStd, "HighVolChangeReturnStd"),
        (LowVolChangeReturnStd, "LowVolChangeReturnStd"),
        (HighVolChangeAmplitude, "HighVolChangeAmplitude"),
        (LowVolChangeAmplitude, "LowVolChangeAmplitude"),
    ]
    
    print("="*60)
    print("放量缩量切分因子测试")
    print("="*60)
    
    all_passed = True
    
    # 基础测试
    print("\n" + "="*60)
    print("第一阶段: 基础功能测试")
    print("="*60)
    
    for FactorClass, name in factors_to_test:
        try:
            test_factor_basic(FactorClass, name)
        except Exception as e:
            print(f"\n[FAIL] {name} 基础测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 真实数据测试
    print("\n" + "="*60)
    print("第二阶段: 真实数据测试")
    print("="*60)
    
    for FactorClass, name in factors_to_test:
        try:
            test_factor_with_real_data(FactorClass, name)
        except Exception as e:
            print(f"\n[FAIL] {name} 真实数据测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 总结
    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查错误信息")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
