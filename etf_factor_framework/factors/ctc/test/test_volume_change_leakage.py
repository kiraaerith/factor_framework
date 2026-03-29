"""
放量缩量切分因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from factors.ctc.volume_change_split import (
    HighVolChangeReturnSum,
    LowVolChangeReturnSum,
    HighVolChangeReturnStd,
    LowVolChangeReturnStd,
    HighVolChangeAmplitude,
    LowVolChangeAmplitude,
)
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
    
    print(f"数据形状: {ohlcv.shape}")
    print(f"日期范围: {ohlcv.dates[0]} 至 {ohlcv.dates[-1]}")
    
    factors_to_test = [
        (HighVolChangeReturnSum, "HighVolChangeReturnSum"),
        (LowVolChangeReturnSum, "LowVolChangeReturnSum"),
        (HighVolChangeReturnStd, "HighVolChangeReturnStd"),
        (LowVolChangeReturnStd, "LowVolChangeReturnStd"),
        (HighVolChangeAmplitude, "HighVolChangeAmplitude"),
        (LowVolChangeAmplitude, "LowVolChangeAmplitude"),
    ]
    
    all_passed = True
    
    for FactorClass, name in factors_to_test:
        print("\n" + "="*60)
        print(f"数据泄露检测: {name}")
        print("="*60)
        
        # 创建因子
        factor = FactorClass(window=20, top_pct=0.2)
        
        # 使用便捷函数检测
        try:
            report = detect_leakage(factor, ohlcv, split_ratio=0.5, verbose=False)
            
            print(f"\n检测结果:")
            print(f"  是否有泄露: {report.has_leakage}")
            print(f"  不匹配比例: {report.mismatch_ratio:.4%}")
            print(f"  最大差异: {report.max_absolute_diff:.8f}")
            
            if report.has_leakage:
                print(f"\n[WARNING] {name} 存在未来数据泄露!")
                all_passed = False
            else:
                print(f"\n[PASS] {name} 无数据泄露，可以继续测评流程")
        except Exception as e:
            print(f"\n[ERROR] {name} 检测失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 总结
    print("\n" + "="*60)
    if all_passed:
        print("所有因子通过数据泄露检测！")
    else:
        print("部分因子存在数据泄露或检测失败，请检查")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
