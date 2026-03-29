"""
HighPriceVolumeChange 因子数据泄露检测脚本 (简化版)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from core.ohlcv_data import OHLCVData
from factors.ctc.price_volume_split import HighPriceVolumeChange


def detect_leakage_simple(factor, ohlcv_data, split_ratio=0.5):
    """简化的泄露检测"""
    from core.ohlcv_data import OHLCVData
    
    n_periods = ohlcv_data.n_periods
    split_point = int(n_periods * split_ratio)
    dates = ohlcv_data.dates
    
    # 前半部分数据 - 手动切片
    short_dates = dates[:split_point]
    short_data = OHLCVData(
        open=ohlcv_data.open[short_dates],
        high=ohlcv_data.high[short_dates],
        low=ohlcv_data.low[short_dates],
        close=ohlcv_data.close[short_dates],
        volume=ohlcv_data.volume[short_dates]
    )
    # 完整数据
    full_data = ohlcv_data
    
    print(f"Short data periods: {short_data.n_periods}")
    print(f"Full data periods: {full_data.n_periods}")
    
    # 计算因子
    print("\nCalculating factor on short data...")
    short_factor = factor.calculate(short_data)
    
    print("Calculating factor on full data...")
    full_factor = factor.calculate(full_data)
    
    # 比较重叠部分
    short_values = short_factor.values
    full_values = full_factor.values.iloc[:, :split_point]
    
    # 对齐比较
    comparison = short_values.values - full_values.values
    valid_mask = ~(np.isnan(short_values.values) | np.isnan(full_values.values))
    
    if valid_mask.sum() == 0:
        print("No valid data for comparison")
        return False
    
    diff = np.abs(comparison[valid_mask])
    max_diff = diff.max()
    mean_diff = diff.mean()
    mismatch_ratio = (diff > 1e-10).sum() / len(diff)
    
    print(f"\n=== Leakage Detection Report ===")
    print(f"Max absolute diff: {max_diff:.10f}")
    print(f"Mean absolute diff: {mean_diff:.10f}")
    print(f"Mismatch ratio: {mismatch_ratio:.4%}")
    
    # 判断标准
    has_leakage = max_diff > 1e-6 or mismatch_ratio > 0.01
    
    if has_leakage:
        print(f"RESULT: LEAKAGE DETECTED!")
        return False
    else:
        print(f"RESULT: PASSED - No leakage detected")
        return True


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
    
    print(f"Data shape: {ohlcv.shape}")
    print(f"Date range: {ohlcv.dates[0]} to {ohlcv.dates[-1]}")
    
    # 创建因子
    factor = HighPriceVolumeChange(window=20, top_pct=0.2)
    print(f"\nFactor: {factor.name}")
    
    # 检测泄露
    success = detect_leakage_simple(factor, ohlcv, split_ratio=0.5)
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
