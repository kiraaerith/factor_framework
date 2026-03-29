"""
动量因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from momentum_factors import MomentumFactor
from factors.leakage_detector import LeakageDetector, detect_leakage


def main():
    # 加载数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'etf_rotation_daily.csv'
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
    
    # 测试多个参数组合
    test_params = [
        {'offset': 0, 'lookback': 20},
        {'offset': 5, 'lookback': 55},
        {'offset': 20, 'lookback': 144},
    ]
    
    all_passed = True
    
    for params in test_params:
        print(f"\n{'='*60}")
        print(f"Testing MomentumFactor with params: {params}")
        print(f"{'='*60}")
        
        # 创建因子
        factor = MomentumFactor(**params)
        
        # 使用检测器类（不使用verbose避免编码问题）
        detector = LeakageDetector(split_ratio=0.5, verbose=False)
        report = detector.detect(factor, ohlcv)
        
        # 手动打印报告（避免emoji）
        print(f"Factor: {report.factor_name}")
        print(f"Result: {'LEAKAGE DETECTED' if report.has_leakage else 'NO LEAKAGE'}")
        print(f"Split ratio: {report.split_ratio:.1%}")
        print(f"Short data periods: {report.short_data_periods}")
        print(f"Full data periods: {report.full_data_periods}")
        print(f"Overlap periods: {report.overlap_periods}")
        print(f"Mismatched periods: {report.mismatched_periods}")
        print(f"Mismatch ratio: {report.mismatch_ratio:.4%}")
        print(f"Max absolute diff: {report.max_absolute_diff:.8f}")
        
        # 检查结果
        if report.has_leakage:
            print(f"[FAIL] {factor.name} has future data leakage!")
            all_passed = False
        else:
            print(f"[PASS] {factor.name} has NO data leakage")
    
    # 最终总结
    print(f"\n{'='*60}")
    print("Final Result")
    print(f"{'='*60}")
    if all_passed:
        print("All momentum factor variants PASSED leakage detection!")
        return True
    else:
        print("Some momentum factor variants FAILED leakage detection!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
