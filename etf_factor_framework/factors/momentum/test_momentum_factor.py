"""
动量因子单元测试脚本
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.ohlcv_data import OHLCVData
from momentum_factors import MomentumFactor


def create_test_data(n_assets: int = 5, n_periods: int = 200) -> OHLCVData:
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


def test_factor_basic():
    """测试因子基础功能"""
    print("="*60)
    print("测试1: 因子基础功能")
    print("="*60)
    
    # 准备数据
    ohlcv = create_test_data()
    print(f"数据形状: {ohlcv.shape}")
    print(f"ETF数量: {ohlcv.n_assets}")
    print(f"时间长度: {ohlcv.n_periods}")
    
    # 创建因子
    factor = MomentumFactor(offset=0, lookback=20)
    print(f"\n因子名称: {factor.name}")
    print(f"因子类型: {factor.factor_type}")
    print(f"因子参数: {factor.params}")
    
    # 计算因子
    print("\n计算因子...")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    # 检查输出格式 (ohlcv.shape 是 (N, T, 5)，因子 shape 是 (N, T))
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods), "因子形状与数据不匹配"
    assert list(factor_data.values.index) == list(ohlcv.symbols), "symbol索引不匹配"
    assert list(factor_data.values.columns) == list(ohlcv.dates), "日期列不匹配"
    
    print("\n✅ 基础功能测试通过")


def test_factor_variants():
    """测试不同参数组合的因子"""
    print("\n" + "="*60)
    print("测试2: 不同参数组合")
    print("="*60)
    
    ohlcv = create_test_data(n_periods=200)
    
    # 测试不同参数组合
    test_params = [
        {'offset': 0, 'lookback': 2},
        {'offset': 0, 'lookback': 20},
        {'offset': 5, 'lookback': 20},
        {'offset': 10, 'lookback': 55},
        {'offset': 20, 'lookback': 144},
    ]
    
    for params in test_params:
        factor = MomentumFactor(**params)
        factor_data = factor.calculate(ohlcv)
        
        # 检查NaN值数量是否合理（前offset+lookback个应为NaN）
        nan_count = factor_data.values.isna().sum().sum()
        expected_nan = ohlcv.n_assets * (params['offset'] + params['lookback'])
        
        print(f"\n参数 {params}:")
        print(f"  名称: {factor.name}")
        print(f"  NaN数量: {nan_count} (预期约: {expected_nan})")
        print(f"  因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
        
        # 检查NaN值位置是否正确
        first_valid_idx = params['offset'] + params['lookback']
        for symbol in ohlcv.symbols:
            row = factor_data.values.loc[symbol]
            nan_in_row = row.iloc[:first_valid_idx].isna().sum()
            non_nan_in_valid = row.iloc[first_valid_idx:].isna().sum()
            assert nan_in_row == first_valid_idx, f"前{first_valid_idx}个应为NaN"
            assert non_nan_in_valid == 0, f"有效数据部分不应有NaN"
    
    print("\n✅ 不同参数组合测试通过")


def test_factor_with_real_data():
    """使用真实数据测试因子"""
    print("\n" + "="*60)
    print("测试3: 真实数据测试")
    print("="*60)
    
    # 加载真实数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'etf_rotation_daily.csv'
    )
    
    if not os.path.exists(csv_path):
        print(f"⚠️ 数据文件不存在: {csv_path}")
        print("跳过真实数据测试")
        return
    
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
    
    # 测试多个参数组合
    test_params = [
        {'offset': 0, 'lookback': 20},
        {'offset': 5, 'lookback': 55},
        {'offset': 20, 'lookback': 144},
    ]
    
    for params in test_params:
        factor = MomentumFactor(**params)
        factor_data = factor.calculate(ohlcv)
        
        print(f"\n参数 {params}:")
        print(f"  因子形状: {factor_data.shape}")
        print(f"  均值: {factor_data.values.mean().mean():.4f}")
        print(f"  标准差: {factor_data.values.std().mean():.4f}")
        print(f"  最小值: {factor_data.values.min().min():.4f}")
        print(f"  最大值: {factor_data.values.max().max():.4f}")
    
    print("\n✅ 真实数据测试通过")


if __name__ == '__main__':
    test_factor_basic()
    test_factor_variants()
    test_factor_with_real_data()
    print("\n" + "="*60)
    print("所有测试通过！")
    print("="*60)
