"""
不平衡度因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from factors.ctc.volume_price_imbalance import VolAmplitudeImbalance, VolReturnStdImbalance
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
    
    # 测试 VolAmplitudeImbalance
    print("\n" + "="*60)
    print("测试因子1: VolAmplitudeImbalance")
    print("="*60)
    
    factor1 = VolAmplitudeImbalance(window=20, top_pct=0.2)
    detector1 = LeakageDetector(split_ratio=0.5, verbose=False)
    report1 = detector1.detect(factor1, ohlcv)
    
    print("\n" + "-"*60)
    print(f"检测结果: {'[FAIL] 存在泄露' if report1.has_leakage else '[PASS] 无泄露'}")
    print(f"不匹配比例: {report1.mismatch_ratio:.4%}")
    print(f"最大差异: {report1.max_absolute_diff:.8f}")
    if report1.has_leakage:
        result1 = False
    else:
        print(f"[PASS] {factor1.name} 无数据泄露，可以继续测评流程")
        result1 = True
    
    # 测试 VolReturnStdImbalance
    print("\n" + "="*60)
    print("测试因子2: VolReturnStdImbalance")
    print("="*60)
    
    factor2 = VolReturnStdImbalance(window=20, top_pct=0.2)
    detector2 = LeakageDetector(split_ratio=0.5, verbose=False)
    report2 = detector2.detect(factor2, ohlcv)
    
    print("\n" + "-"*60)
    print(f"检测结果: {'[FAIL] 存在泄露' if report2.has_leakage else '[PASS] 无泄露'}")
    print(f"不匹配比例: {report2.mismatch_ratio:.4%}")
    print(f"最大差异: {report2.max_absolute_diff:.8f}")
    if report2.has_leakage:
        result2 = False
    else:
        print(f"[PASS] {factor2.name} 无数据泄露，可以继续测评流程")
        result2 = True
    
    # 总结
    print("\n" + "="*60)
    print("检测结果总结")
    print("="*60)
    if result1 and result2:
        print("[PASS] 所有因子均无数据泄露，可以继续测评流程")
        return True
    else:
        print("[FAIL] 部分因子存在数据泄露，需要修复因子代码")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
