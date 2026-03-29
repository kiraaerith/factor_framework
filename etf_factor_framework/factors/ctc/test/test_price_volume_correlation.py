"""
PriceVolumeCorrelation 因子单元测试脚本
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.ohlcv_data import OHLCVData
from factors.ctc.price_volume_correlation import PVCorr, DPVCorr, PdVCorr, DPdVCorr


def create_test_data(n_assets: int = 5, n_periods: int = 100) -> OHLCVData:
    """创建测试数据"""
    symbols = [f'ETF{i}' for i in range(n_assets)]
    dates = pd.date_range('2024-01-01', periods=n_periods)
    
    # 生成随机价格数据
    np.random.seed(42)
    close = pd.DataFrame(
        np.random.randn(n_assets, n_periods).cumsum(axis=1) + 100,
        index=symbols, columns=dates
    )
    
    # 生成与价格有一定相关性的成交量
    volume_base = np.abs(np.random.randn(n_assets, n_periods)) * 1000
    volume_price_effect = np.abs(close.values - 100) * 10  # 价格偏离100越多，成交量越大
    volume = pd.DataFrame(
        volume_base + volume_price_effect,
        index=symbols, columns=dates
    )
    
    ohlcv = OHLCVData(
        open=close * 0.99,
        high=close * 1.02,
        low=close * 0.98,
        close=close,
        volume=volume
    )
    return ohlcv


def test_pv_corr():
    """测试PVCorr因子"""
    print("="*60)
    print("测试1: PVCorr因子 (收盘价与成交量相关系数)")
    print("="*60)
    
    # 准备数据
    ohlcv = create_test_data()
    print(f"数据形状: {ohlcv.shape}")
    
    # 创建因子
    factor = PVCorr(window=20)
    print(f"因子名称: {factor.name}")
    print(f"因子类型: {factor.factor_type}")
    print(f"因子参数: {factor.params}")
    
    # 计算因子
    print("\n计算因子...")
    factor_data = factor.calculate(ohlcv)
    
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    # 检查相关系数范围
    valid_values = factor_data.values.dropna()
    if len(valid_values) > 0:
        assert valid_values.min() >= -1.0 and valid_values.max() <= 1.0, "相关系数应在[-1, 1]范围内"
    
    # 检查输出格式
    assert factor_data.values.shape == (ohlcv.n_assets, ohlcv.n_periods), f"因子形状{factor_data.values.shape}与数据({ohlcv.n_assets}, {ohlcv.n_periods})不匹配"
    assert list(factor_data.values.index) == list(ohlcv.symbols), "symbol索引不匹配"
    assert list(factor_data.values.columns) == list(ohlcv.dates), "日期列不匹配"
    
    print("\n[OK] PVCorr测试通过")


def test_dpv_corr():
    """测试DPVCorr因子"""
    print("\n" + "="*60)
    print("测试2: DPVCorr因子 (收益率与成交量相关系数)")
    print("="*60)
    
    ohlcv = create_test_data()
    factor = DPVCorr(window=20)
    print(f"因子名称: {factor.name}")
    
    factor_data = factor.calculate(ohlcv)
    print(f"因子形状: {factor_data.shape}")
    print(f"因子值范围: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    print(f"NaN值数量: {factor_data.values.isna().sum().sum()}")
    
    print("\n[OK] DPVCorr测试通过")


def test_lag_versions():
    """测试领先滞后版本"""
    print("\n" + "="*60)
    print("测试3: 领先滞后版本")
    print("="*60)
    
    ohlcv = create_test_data()
    
    # 测试lag1（滞后1期）
    factor_lag1 = PVCorr(window=20, lag=1)
    print(f"lag=1因子名称: {factor_lag1.name}")
    factor_data_lag1 = factor_lag1.calculate(ohlcv)
    print(f"lag=1因子NaN值数量: {factor_data_lag1.values.isna().sum().sum()}")
    
    # 测试lead1（领先1期）
    factor_lead1 = PVCorr(window=20, lag=-1)
    print(f"lead=1因子名称: {factor_lead1.name}")
    factor_data_lead1 = factor_lead1.calculate(ohlcv)
    print(f"lead=1因子NaN值数量: {factor_data_lead1.values.isna().sum().sum()}")
    
    print("\n[OK] 领先滞后版本测试通过")


def test_with_real_data():
    """使用真实数据测试因子"""
    print("\n" + "="*60)
    print("测试4: 真实数据测试")
    print("="*60)
    
    # 加载真实数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    if not os.path.exists(csv_path):
        print(f"⚠ 数据文件不存在: {csv_path}")
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
    
    # 测试各类型因子
    for FactorClass, name in [(PVCorr, 'PV'), (DPVCorr, 'DPV'), (PdVCorr, 'PdV'), (DPdVCorr, 'DPdV')]:
        factor = FactorClass(window=20)
        factor_data = factor.calculate(ohlcv)
        # 使用values.flatten()获取所有值，然后过滤NaN
        all_values = factor_data.values.values.flatten()
        valid_values = all_values[~np.isnan(all_values)]
        print(f"\n{name}Corr:")
        print(f"  - 有效值数量: {len(valid_values)}")
        print(f"  - 均值: {np.mean(valid_values):.4f}")
        print(f"  - 标准差: {np.std(valid_values):.4f}")
    
    print("\n[OK] 真实数据测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("PriceVolumeCorrelation 因子测试套件")
    print("="*60)
    
    try:
        test_pv_corr()
        test_dpv_corr()
        test_lag_versions()
        test_with_real_data()
        
        print("\n" + "="*60)
        print("所有测试通过！")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
