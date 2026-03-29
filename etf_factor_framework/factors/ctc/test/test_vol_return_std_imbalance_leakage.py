"""
VolReturnStdImbalance 因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from factors.ctc.volume_price_imbalance import VolReturnStdImbalance
from factors.leakage_detector import LeakageDetector, detect_leakage


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
    factor = VolReturnStdImbalance(window=20, top_pct=0.2)
    
    # 方法1: 使用便捷函数检测
    print("\n" + "="*60)
    print("Data Leakage Detection (using convenience function)")
    print("="*60)
    report = detect_leakage(factor, ohlcv, split_ratio=0.5, verbose=False)
    
    # 检查结果
    print("\n" + "="*60)
    print("Detection Result Summary")
    print("="*60)
    if report.has_leakage:
        print(f"[WARNING] {factor.name} has future data leakage!")
        print(f"   Mismatch ratio: {report.mismatch_ratio:.4%}")
        print(f"   Max absolute diff: {report.max_absolute_diff:.8f}")
        return False
    else:
        print(f"[PASS] {factor.name} has NO data leakage")
        print(f"   Mismatch ratio: {report.mismatch_ratio:.4%}")
        print(f"   Max absolute diff: {report.max_absolute_diff:.8f}")
        print("Can proceed with evaluation")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
