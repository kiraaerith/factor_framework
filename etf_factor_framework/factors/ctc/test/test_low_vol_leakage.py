"""
LowVolReturnSum 因子数据泄露验证脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from core.ohlcv_data import OHLCVData
from factors.ctc.volume_price_split import LowVolReturnSum


def verify_no_leakage():
    """验证因子无数据泄露"""
    print("="*60)
    print("LowVolReturnSum 数据泄露验证")
    print("="*60)
    
    # 加载数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    df = pd.read_csv(csv_path)
    df['eob'] = pd.to_datetime(df['eob'])
    df.columns = [c.lower() for c in df.columns]
    
    ohlcv_full = OHLCVData.from_dataframe(
        df,
        symbol_col='symbol',
        date_col='eob',
        ohlcv_cols={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    )
    
    print(f"\n完整数据形状: ({ohlcv_full.n_assets}, {ohlcv_full.n_periods})")
    
    # 取前50%数据作为截断数据
    n_periods = ohlcv_full.n_periods
    split_point = n_periods // 2
    
    ohlcv_truncated = OHLCVData(
        open=ohlcv_full.open.iloc[:, :split_point],
        high=ohlcv_full.high.iloc[:, :split_point],
        low=ohlcv_full.low.iloc[:, :split_point],
        close=ohlcv_full.close.iloc[:, :split_point],
        volume=ohlcv_full.volume.iloc[:, :split_point]
    )
    
    # 创建因子
    window = 20
    factor = LowVolReturnSum(window=window, top_pct=0.2)
    
    print(f"因子参数: window={window}, top_pct=0.2")
    print(f"因子名称: {factor.name}")
    
    # 计算两个数据集的因子值
    print("\n计算完整数据集的因子值...")
    factor_full = factor.calculate(ohlcv_full)
    
    print("计算截断数据集的因子值...")
    factor_truncated = factor.calculate(ohlcv_truncated)
    
    # 对比截断点前的数据
    compare_end = split_point
    full_values = factor_full.values.iloc[:, window-1:compare_end]
    truncated_values = factor_truncated.values.iloc[:, window-1:]
    
    print(f"\n对比区间: 第{window}期 至 第{compare_end}期")
    
    # 计算差异
    diff = (full_values.values - truncated_values.values)
    abs_diff = np.abs(diff)
    max_diff = np.nanmax(abs_diff)
    mean_diff = np.nanmean(abs_diff)
    
    print(f"差异统计:")
    print(f"  - 最大绝对差异: {max_diff:.10f}")
    print(f"  - 平均绝对差异: {mean_diff:.10f}")
    
    # 判断是否存在泄露
    threshold = 1e-10
    has_leakage = max_diff > threshold
    
    print("\n" + "="*60)
    if has_leakage:
        print("[!] 警告: 可能存在数据泄露!")
        return False
    else:
        print("[OK] 验证通过: 无数据泄露")
        return True


if __name__ == '__main__':
    success = verify_no_leakage()
    sys.exit(0 if success else 1)
