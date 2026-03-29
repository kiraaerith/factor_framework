"""
LowVolChangeReturnSum 因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from factors.ctc.volume_change_split import LowVolChangeReturnSum
from factors.leakage_detector import LeakageDetector


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
    factor = LowVolChangeReturnSum(window=20, top_pct=0.2)
    
    # 使用检测器类（避免Unicode输出问题）
    print("\n" + "="*60)
    print("Data Leakage Detection")
    print("="*60)
    
    detector = LeakageDetector(split_ratio=0.5, verbose=False)
    report = detector.detect(factor, ohlcv)
    
    # 手动输出报告（避免Unicode字符）
    print(f"\nFactor: {report.factor_name}")
    print(f"Short data periods: {report.short_data_periods}")
    print(f"Full data periods: {report.full_data_periods}")
    print(f"Overlap periods: {report.overlap_periods}")
    print(f"Mismatched periods: {report.mismatched_periods}")
    print(f"Mismatch ratio: {report.mismatch_ratio:.4%}")
    
    if report.has_leakage:
        print(f"[WARNING] Data leakage detected!")
        print(f"   Max absolute diff: {report.max_absolute_diff:.8f}")
        print(f"   Max relative diff: {report.max_relative_diff:.8f}")
        return False
    else:
        print(f"[OK] No data leakage detected. Ready for evaluation.")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
