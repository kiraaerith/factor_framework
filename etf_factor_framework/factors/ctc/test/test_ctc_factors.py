"""
CTC因子单元测试脚本

测试高低成交量切分因子的基础功能和正确性。
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.ohlcv_data import OHLCVData
from factors.ctc.volume_price_split import (
    HighVolReturnSum, LowVolReturnSum,
    HighVolReturnStd, LowVolReturnStd,
    HighVolAmplitude, LowVolAmplitude
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
            np.abs(np.random.randn(n_assets, n_periods)) * 1000 + 1000,
            index=symbols, columns=dates
        )
    )
    return ohlcv


def test_high_vol_return_sum():
    """测试高成交量日收益率之和因子"""
    print("="*60)
    print("测试: HighVolReturnSum 因子")
    print("="*60)
    
    # 准备数据
    ohlcv = create_test_data()
    print(f"数据形状: ({ohlcv.n_assets}, {ohlcv.n_periods})")
    print(f"ETF数量: {ohlcv.n_assets}")
    print(f"时间长度: {ohlcv.n_periods}")
    
    # 创建因子
    factor = HighVolReturnSum(window=20, top_pct=0.2)
    print(f"\n因子名称: {factor.name}")
    print(f"因子类型: {factor.factor_type}")
    print(f"因子参数: {factor.params}")
    
    # 计算因子
    print("\n计算因子...")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"因子值均值: {factor_data.values.mean().mean():.4f}")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    # 验证输出格式 (FactorData.shape返回(n_assets, n_periods)，OHLCVData.shape返回(n_assets, n_periods, 5))
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods), "因子形状与数据不匹配"
    assert list(factor_data.values.index) == list(ohlcv.symbols), "symbol索引不匹配"
    assert list(factor_data.values.columns) == list(ohlcv.dates), "日期列不匹配"
    
    # 验证前window-1期为NaN
    assert factor_data.values.iloc[:, :19].isna().all().all(), "前window-1期应该是NaN"
    
    print("\n[OK] HighVolReturnSum 测试通过")


def test_low_vol_return_sum():
    """测试低成交量日收益率之和因子"""
    print("\n" + "="*60)
    print("测试: LowVolReturnSum 因子")
    print("="*60)
    
    ohlcv = create_test_data()
    factor = LowVolReturnSum(window=20, top_pct=0.2)
    
    print(f"因子名称: {factor.name}")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods)
    print("\n[OK] LowVolReturnSum 测试通过")


def test_high_vol_return_std():
    """测试高成交量日收益率标准差因子"""
    print("\n" + "="*60)
    print("测试: HighVolReturnStd 因子")
    print("="*60)
    
    ohlcv = create_test_data()
    factor = HighVolReturnStd(window=20, top_pct=0.2)
    
    print(f"因子名称: {factor.name}")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods)
    print("\n[OK] HighVolReturnStd 测试通过")


def test_low_vol_return_std():
    """测试低成交量日收益率标准差因子"""
    print("\n" + "="*60)
    print("测试: LowVolReturnStd 因子")
    print("="*60)
    
    ohlcv = create_test_data()
    factor = LowVolReturnStd(window=20, top_pct=0.2)
    
    print(f"因子名称: {factor.name}")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods)
    print("\n[OK] LowVolReturnStd 测试通过")


def test_high_vol_amplitude():
    """测试高成交量日振幅均值因子"""
    print("\n" + "="*60)
    print("测试: HighVolAmplitude 因子")
    print("="*60)
    
    ohlcv = create_test_data()
    factor = HighVolAmplitude(window=20, top_pct=0.2)
    
    print(f"因子名称: {factor.name}")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods)
    print("\n[OK] HighVolAmplitude 测试通过")


def test_low_vol_amplitude():
    """测试低成交量日振幅均值因子"""
    print("\n" + "="*60)
    print("测试: LowVolAmplitude 因子")
    print("="*60)
    
    ohlcv = create_test_data()
    factor = LowVolAmplitude(window=20, top_pct=0.2)
    
    print(f"因子名称: {factor.name}")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    assert factor_data.shape == (ohlcv.n_assets, ohlcv.n_periods)
    print("\n[OK] LowVolAmplitude 测试通过")


def test_with_real_data():
    """使用真实数据测试因子"""
    print("\n" + "="*60)
    print("测试: 真实数据测试")
    print("="*60)
    
    # 加载真实数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
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
    
    # 测试主因子
    factor = HighVolReturnSum(window=20, top_pct=0.2)
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子计算完成，形状: {factor_data.shape}")
    print(f"因子值统计:")
    print(f"  - 均值: {factor_data.values.mean().mean():.6f}")
    print(f"  - 标准差: {factor_data.values.std().mean():.6f}")
    print(f"  - 最小值: {factor_data.values.min().min():.6f}")
    print(f"  - 最大值: {factor_data.values.max().max():.6f}")
    print(f"  - NaN比例: {factor_data.values.isna().mean().mean():.2%}")
    
    print("\n[OK] 真实数据测试通过")


def test_parameter_validation():
    """测试参数验证"""
    print("\n" + "="*60)
    print("测试: 参数验证")
    print("="*60)
    
    # 测试无效参数
    try:
        HighVolReturnSum(window=-1, top_pct=0.2)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"[OK] window=-1 正确抛出错误: {e}")
    
    try:
        HighVolReturnSum(window=20, top_pct=0.8)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"[OK] top_pct=0.8 正确抛出错误: {e}")
    
    try:
        HighVolReturnSum(window=20, top_pct=-0.1)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"[OK] top_pct=-0.1 正确抛出错误: {e}")
    
    print("\n[OK] 参数验证测试通过")


if __name__ == '__main__':
    test_high_vol_return_sum()
    test_low_vol_return_sum()
    test_high_vol_return_std()
    test_low_vol_return_std()
    test_high_vol_amplitude()
    test_low_vol_amplitude()
    test_with_real_data()
    test_parameter_validation()
    
    print("\n" + "="*60)
    print("[DONE] 所有测试通过！")
    print("="*60)
