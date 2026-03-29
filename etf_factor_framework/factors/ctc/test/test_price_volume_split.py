"""
CTC量价因子 - 高低价区间切分因子测试脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from core.ohlcv_data import OHLCVData
from factors.ctc.price_volume_split import (
    HighPriceRelativeVolume,
    LowPriceRelativeVolume,
    HighPriceVolumeChange,
    LowPriceVolumeChange,
)


def create_test_data(n_assets: int = 5, n_periods: int = 60) -> OHLCVData:
    """创建测试数据"""
    symbols = [f'ETF{i}' for i in range(n_assets)]
    dates = pd.date_range('2024-01-01', periods=n_periods)
    
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


def test_high_price_relative_volume():
    """测试高价日相对成交量因子"""
    print("="*60)
    print("测试: HighPriceRelativeVolume")
    print("="*60)
    
    ohlcv = create_test_data()
    print(f"数据形状: {ohlcv.shape}")
    
    factor = HighPriceRelativeVolume(window=20, top_pct=0.2)
    print(f"因子名称: {factor.name}")
    print(f"因子类型: {factor.factor_type}")
    print(f"因子参数: {factor.params}")
    
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    # 验证因子值应该大于0（相对成交量）
    valid_values = factor_data.values.dropna()
    assert (valid_values > 0).all().item(), "相对成交量应该大于0"
    
    print("[PASS] HighPriceRelativeVolume 测试通过")
    return True


def test_low_price_relative_volume():
    """测试低价日相对成交量因子"""
    print("\n" + "="*60)
    print("测试: LowPriceRelativeVolume")
    print("="*60)
    
    ohlcv = create_test_data()
    
    factor = LowPriceRelativeVolume(window=20, top_pct=0.2)
    print(f"因子名称: {factor.name}")
    
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    
    valid_values = factor_data.values.dropna()
    assert (valid_values > 0).all().item(), "相对成交量应该大于0"
    
    print("[PASS] LowPriceRelativeVolume 测试通过")
    return True


def test_high_price_volume_change():
    """测试高价日成交量变化因子"""
    print("\n" + "="*60)
    print("测试: HighPriceVolumeChange")
    print("="*60)
    
    ohlcv = create_test_data()
    
    factor = HighPriceVolumeChange(window=20, top_pct=0.2)
    print(f"因子名称: {factor.name}")
    
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    
    valid_values = factor_data.values.dropna()
    assert (valid_values >= 0).all().item(), "成交量变化比值应该大于等于0"
    
    print("[PASS] HighPriceVolumeChange 测试通过")
    return True


def test_low_price_volume_change():
    """测试低价日成交量变化因子"""
    print("\n" + "="*60)
    print("测试: LowPriceVolumeChange")
    print("="*60)
    
    ohlcv = create_test_data()
    
    factor = LowPriceVolumeChange(window=20, top_pct=0.2)
    print(f"因子名称: {factor.name}")
    
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    
    valid_values = factor_data.values.dropna()
    assert (valid_values >= 0).all().item(), "成交量变化比值应该大于等于0"
    
    print("[PASS] LowPriceVolumeChange 测试通过")
    return True


def test_with_real_data():
    """使用真实数据测试"""
    print("\n" + "="*60)
    print("测试: 使用真实数据")
    print("="*60)
    
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    if not os.path.exists(csv_path):
        print(f"[WARN] 数据文件不存在: {csv_path}")
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
    
    # 测试 HighPriceRelativeVolume
    factor = HighPriceRelativeVolume(window=20, top_pct=0.2)
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子计算完成，形状: {factor_data.shape}")
    print(f"因子值统计:")
    print(f"  - 均值: {factor_data.values.mean().mean():.4f}")
    print(f"  - 标准差: {factor_data.values.std().mean():.4f}")
    print(f"  - 最小值: {factor_data.values.min().min():.4f}")
    print(f"  - 最大值: {factor_data.values.max().max():.4f}")
    
    print("[PASS] 真实数据测试通过")
    return True


def test_economic_logic():
    """测试经济学逻辑"""
    print("\n" + "="*60)
    print("测试: 经济学逻辑验证")
    print("="*60)
    
    # 构造特定数据：高价日成交量明显高于平均
    symbols = ['ETF1', 'ETF2']
    dates = pd.date_range('2024-01-01', periods=20)
    
    # ETF1: 高价日成交量高
    close1 = pd.DataFrame([[100 + i for i in range(20)], [100 - i for i in range(20)]], 
                          index=symbols, columns=dates)
    # 成交量与价格正相关
    volume1 = pd.DataFrame([[1000 + i*100 for i in range(20)], [1000 - i*50 for i in range(20)]],
                           index=symbols, columns=dates)
    
    ohlcv = OHLCVData(
        open=close1 * 0.99,
        high=close1 * 1.01,
        low=close1 * 0.99,
        close=close1,
        volume=volume1
    )
    
    factor = HighPriceRelativeVolume(window=20, top_pct=0.2)
    factor_data = factor.calculate(ohlcv)
    
    # ETF1的高价日相对成交量应该大于1（因为高价日成交量更高）
    etf1_values = factor_data.values.loc['ETF1'].dropna()
    print(f"ETF1 (高价日成交量高) 的相对成交量均值: {etf1_values.mean():.4f}")
    
    # ETF2的高价日相对成交量应该小于1（因为高价日成交量更低）
    etf2_values = factor_data.values.loc['ETF2'].dropna()
    print(f"ETF2 (高价日成交量低) 的相对成交量均值: {etf2_values.mean():.4f}")
    
    assert etf1_values.mean() > 1.0, "高价日成交量高的股票，相对成交量应大于1"
    assert etf2_values.mean() < 1.0, "高价日成交量低的股票，相对成交量应小于1"
    
    print("[PASS] 经济学逻辑验证通过")
    return True


if __name__ == '__main__':
    all_passed = True
    
    try:
        test_high_price_relative_volume()
    except Exception as e:
        print(f"[FAIL] HighPriceRelativeVolume 测试失败: {e}")
        all_passed = False
    
    try:
        test_low_price_relative_volume()
    except Exception as e:
        print(f"[FAIL] LowPriceRelativeVolume 测试失败: {e}")
        all_passed = False
    
    try:
        test_high_price_volume_change()
    except Exception as e:
        print(f"[FAIL] HighPriceVolumeChange 测试失败: {e}")
        all_passed = False
    
    try:
        test_low_price_volume_change()
    except Exception as e:
        print(f"[FAIL] LowPriceVolumeChange 测试失败: {e}")
        all_passed = False
    
    try:
        test_with_real_data()
    except Exception as e:
        print(f"[FAIL] 真实数据测试失败: {e}")
        all_passed = False
    
    try:
        test_economic_logic()
    except Exception as e:
        print(f"[FAIL] 经济学逻辑测试失败: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查代码")
    print("="*60)
