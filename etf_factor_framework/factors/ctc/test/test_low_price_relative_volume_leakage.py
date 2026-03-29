"""
LowPriceRelativeVolume 因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from core.ohlcv_data import OHLCVData
from factors.ctc.price_volume_split import LowPriceRelativeVolume


def detect_leakage_manual(calculator, ohlcv_data, split_ratio=0.5):
    """
    手动实现泄露检测，避免编码问题
    """
    n_periods = ohlcv_data.n_periods
    split_point = int(n_periods * split_ratio)
    
    # 获取日期列表
    dates = ohlcv_data.dates
    short_dates = dates[:split_point]
    
    # 切割数据
    short_ohlcv = OHLCVData(
        open=ohlcv_data.open[short_dates],
        high=ohlcv_data.high[short_dates],
        low=ohlcv_data.low[short_dates],
        close=ohlcv_data.close[short_dates],
        volume=ohlcv_data.volume[short_dates],
    )
    
    # 使用完整数据计算
    full_ohlcv = ohlcv_data
    
    print(f"分割点: {split_point}")
    print(f"短数据集长度: {short_ohlcv.n_periods}")
    print(f"完整数据集长度: {full_ohlcv.n_periods}")
    
    # 计算因子
    short_factor = calculator.calculate(short_ohlcv)
    full_factor = calculator.calculate(full_ohlcv)
    
    # 获取重叠的时间点
    short_factor_dates = set(short_factor.values.columns)
    full_factor_dates = set(full_factor.values.columns)
    overlap_dates = sorted(list(short_factor_dates & full_factor_dates))
    
    # 获取共同的标的
    short_symbols = set(short_factor.values.index)
    full_symbols = set(full_factor.values.index)
    overlap_symbols = sorted(list(short_symbols & full_symbols))
    
    # 提取重叠部分的数据
    short_values = short_factor.values.loc[overlap_symbols, overlap_dates]
    full_values = full_factor.values.loc[overlap_symbols, overlap_dates]
    
    # 计算差异
    diff = (short_values - full_values).abs()
    max_diff = diff.max().max()
    
    # 计算不匹配比例
    total_elements = short_values.size
    nan_mask = short_values.isna() | full_values.isna()
    comparable_elements = total_elements - nan_mask.sum().sum()
    
    if comparable_elements > 0:
        mismatch_count = (diff > 1e-10).sum().sum()
        mismatch_ratio = mismatch_count / comparable_elements
    else:
        mismatch_ratio = 0
    
    has_leakage = mismatch_ratio > 0.01 or max_diff > 1e-6
    
    return has_leakage, mismatch_ratio, max_diff


def main():
    # 加载数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    df = pd.read_csv(csv_path)
    df['eob'] = pd.to_datetime(df['eob'])
    df.columns = [c.lower() for c in df.columns]
    
    ohlcv = OHLCVData.from_dataframe(
        df,
        symbol_col='symbol',
        date_col='eob',
        ohlcv_cols={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    )
    
    print(f"数据形状: {ohlcv.shape}")
    print(f"日期范围: {ohlcv.dates[0]} 至 {ohlcv.dates[-1]}")
    
    # 创建因子
    factor = LowPriceRelativeVolume(window=20, top_pct=0.2)
    print(f"\n因子名称: {factor.name}")
    
    # 手动泄露检测
    print("\n" + "="*60)
    print("数据泄露检测")
    print("="*60)
    
    has_leakage, mismatch_ratio, max_diff = detect_leakage_manual(
        factor, ohlcv, split_ratio=0.5
    )
    
    print("\n" + "="*60)
    print("检测结果总结")
    print("="*60)
    print(f"因子名称: {factor.name}")
    print(f"是否泄露: {has_leakage}")
    print(f"不匹配比例: {mismatch_ratio:.4%}")
    print(f"最大差异: {max_diff:.8f}")
    
    if has_leakage:
        print(f"[WARNING] {factor.name} 存在未来数据泄露!")
        return False
    else:
        print(f"[OK] {factor.name} 无数据泄露，可以继续测评流程")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
