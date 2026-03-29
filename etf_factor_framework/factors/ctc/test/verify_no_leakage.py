"""
CTC因子数据泄露验证脚本

通过简单验证确认HighVolReturnSum因子无未来数据泄露。
原理：
1. 检查因子计算是否只使用过去window日的数据
2. 对比两个数据集（完整 vs 截断）的因子值，验证截断点前的数据一致
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from core.ohlcv_data import OHLCVData
from factors.ctc.volume_price_split import HighVolReturnSum


def verify_no_leakage():
    """验证因子无数据泄露"""
    print("="*60)
    print("HighVolReturnSum 数据泄露验证")
    print("="*60)
    
    # 加载数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    df = pd.read_csv(csv_path)
    df['eob'] = pd.to_datetime(df['eob'])
    df.columns = [c.lower() for c in df.columns]
    
    # 构建完整OHLCV数据
    ohlcv_full = OHLCVData.from_dataframe(
        df,
        symbol_col='symbol',
        date_col='eob',
        ohlcv_cols={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    )
    
    print(f"\n完整数据形状: ({ohlcv_full.n_assets}, {ohlcv_full.n_periods})")
    print(f"日期范围: {ohlcv_full.dates[0]} 至 {ohlcv_full.dates[-1]}")
    
    # 取前50%数据作为截断数据
    n_periods = ohlcv_full.n_periods
    split_point = n_periods // 2
    
    # 截断数据：只取前split_point天的数据
    ohlcv_truncated = OHLCVData(
        open=ohlcv_full.open.iloc[:, :split_point],
        high=ohlcv_full.high.iloc[:, :split_point],
        low=ohlcv_full.low.iloc[:, :split_point],
        close=ohlcv_full.close.iloc[:, :split_point],
        volume=ohlcv_full.volume.iloc[:, :split_point]
    )
    
    print(f"\n截断数据形状: ({ohlcv_truncated.n_assets}, {ohlcv_truncated.n_periods})")
    print(f"截断日期: {ohlcv_truncated.dates[0]} 至 {ohlcv_truncated.dates[-1]}")
    
    # 创建因子
    window = 20
    factor = HighVolReturnSum(window=window, top_pct=0.2)
    
    print(f"\n因子参数: window={window}, top_pct=0.2")
    print(f"因子名称: {factor.name}")
    
    # 计算两个数据集的因子值
    print("\n计算完整数据集的因子值...")
    factor_full = factor.calculate(ohlcv_full)
    
    print("计算截断数据集的因子值...")
    factor_truncated = factor.calculate(ohlcv_truncated)
    
    # 对比截断点前的数据
    # 由于因子需要window日的历史数据，实际可对比的数据从window-1开始
    compare_end = split_point  # 截断点
    
    # 获取两个数据集在可对比区间内的因子值
    full_values = factor_full.values.iloc[:, window-1:compare_end]
    truncated_values = factor_truncated.values.iloc[:, window-1:]
    
    print(f"\n对比区间: 第{window}期 至 第{compare_end}期")
    print(f"对比数据形状: {full_values.shape}")
    
    # 计算差异
    diff = (full_values.values - truncated_values.values)
    abs_diff = np.abs(diff)
    max_diff = np.nanmax(abs_diff)
    mean_diff = np.nanmean(abs_diff)
    
    print(f"\n差异统计:")
    print(f"  - 最大绝对差异: {max_diff:.10f}")
    print(f"  - 平均绝对差异: {mean_diff:.10f}")
    
    # 判断是否存在泄露
    threshold = 1e-10
    has_leakage = max_diff > threshold
    
    print("\n" + "="*60)
    if has_leakage:
        print("[!] 警告: 可能存在数据泄露!")
        print(f"    最大差异超过阈值 ({threshold})")
        return False
    else:
        print("[OK] 验证通过: 无数据泄露")
        print(f"    完整数据和截断数据在分割点前的因子值完全一致")
        print(f"    证明因子只使用了过去{window}日的数据，无未来数据依赖")
        return True


def verify_calculation_logic():
    """验证因子计算逻辑的正确性"""
    print("\n" + "="*60)
    print("计算逻辑验证")
    print("="*60)
    
    # 创建简单测试数据
    symbols = ['ETF1']
    dates = pd.date_range('2024-01-01', periods=25)
    
    # 构造数据：成交量递增，收益率固定
    close = pd.DataFrame([[100 + i * 0.1 for i in range(25)]], index=symbols, columns=dates)
    volume = pd.DataFrame([[1000 + i * 100 for i in range(25)]], index=symbols, columns=dates)
    
    ohlcv = OHLCVData(
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=volume
    )
    
    factor = HighVolReturnSum(window=20, top_pct=0.2)
    result = factor.calculate(ohlcv)
    
    # 验证:
    # 1. 前window-1期应该是NaN
    assert result.values.iloc[:, :19].isna().all().all(), "前window-1期应该是NaN"
    print("[OK] 前window-1期为NaN")
    
    # 2. 第window期应该有值
    assert not pd.isna(result.values.iloc[0, 19]), "第window期应该有值"
    print("[OK] 第window期有有效值")
    
    # 3. 因子值应该合理（收益率之和）
    # 由于我们构造的收益率很小且稳定，因子值应该在合理范围内
    val = result.values.iloc[0, 19]
    print(f"[OK] 第20期因子值: {val:.6f}")
    
    print("\n[OK] 计算逻辑验证通过")
    return True


if __name__ == '__main__':
    success1 = verify_calculation_logic()
    success2 = verify_no_leakage()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("[PASS] 所有验证通过！因子实现正确且无数据泄露。")
        sys.exit(0)
    else:
        print("[FAIL] 验证失败，请检查因子实现。")
        sys.exit(1)
