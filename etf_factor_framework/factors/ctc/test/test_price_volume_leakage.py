"""
CTC量价因子 - 高低价区间切分因子数据泄露检测
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from factors.ctc.price_volume_split import HighPriceRelativeVolume
from factors.leakage_detector import detect_leakage


def main():
    # 加载数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    if not os.path.exists(csv_path):
        print(f"[WARN] 数据文件不存在: {csv_path}")
        return True
    
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
    factor = HighPriceRelativeVolume(window=20, top_pct=0.2)
    
    # 数据泄露检测
    print("\n" + "="*60)
    print("数据泄露检测")
    print("="*60)
    
    report = detect_leakage(factor, ohlcv, split_ratio=0.5, verbose=False)
    
    # 检查结果
    print("\n" + "="*60)
    print("检测结果总结")
    print("="*60)
    
    print(f"是否有泄露: {report.has_leakage}")
    print(f"不匹配比例: {report.mismatch_ratio:.4%}")
    print(f"最大差异: {report.max_absolute_diff:.8f}")
    
    if report.has_leakage:
        print(f"[FAIL] {factor.name} 存在未来数据泄露!")
        return False
    else:
        print(f"[PASS] {factor.name} 无数据泄露，可以继续测评流程")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
