"""
HighVolAmplitude 因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from factors.ctc.volume_price_split import HighVolAmplitude
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
    
    print(f"数据形状: {ohlcv.shape}")
    print(f"日期范围: {ohlcv.dates[0]} 至 {ohlcv.dates[-1]}")
    
    # 创建因子
    factor = HighVolAmplitude(window=20, top_pct=0.2)
    
    # 使用检测器类检测
    print("\n" + "="*60)
    print("数据泄露检测")
    print("="*60)
    detector = LeakageDetector(split_ratio=0.5, verbose=False)
    report = detector.detect(factor, ohlcv)
    
    # 手动打印报告（避免Unicode问题）
    print("=" * 60)
    print(f"因子数据泄露检测报告: {report.factor_name}")
    print("=" * 60)
    status = "警告: 存在数据泄露" if report.has_leakage else "OK: 无数据泄露"
    print(f"检测结果: {status}")
    print(f"切割比例: {report.split_ratio:.1%}")
    print(f"短数据集长度: {report.short_data_periods} 期")
    print(f"完整数据集长度: {report.full_data_periods} 期")
    print(f"重叠时间段: {report.overlap_periods} 期")
    print("-" * 60)
    print(f"不匹配时间点数: {report.mismatched_periods}")
    print(f"不匹配比例: {report.mismatch_ratio:.4%}")
    print(f"最大绝对差异: {report.max_absolute_diff:.8f}")
    print(f"最大相对差异: {report.max_relative_diff:.4%}")
    print("=" * 60)
    
    # 检查结果
    print("\n" + "="*60)
    print("检测结果总结")
    print("="*60)
    if report.has_leakage:
        print(f"警告: {factor.name} 存在未来数据泄露!")
        print(f"   不匹配比例: {report.mismatch_ratio:.4%}")
        print(f"   最大差异: {report.max_absolute_diff:.8f}")
        return False
    else:
        print(f"[OK] {factor.name} 无数据泄露，可以继续测评流程")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
