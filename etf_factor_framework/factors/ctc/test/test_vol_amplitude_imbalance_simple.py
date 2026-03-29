"""
VolAmplitudeImbalance 因子简单测试脚本
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.ohlcv_data import OHLCVData
from factors.ctc.volume_price_imbalance import VolAmplitudeImbalance
from factors.leakage_detector import LeakageDetector


def main():
    # 加载真实数据
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'etf_rotation_daily.csv'
    )
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df['eob'] = pd.to_datetime(df['eob'])
    df.columns = [c.lower() for c in df.columns]
    
    ohlcv = OHLCVData.from_dataframe(
        df,
        symbol_col='symbol',
        date_col='eob',
        ohlcv_cols={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    )
    
    print(f"Data shape: {ohlcv.n_assets} assets x {ohlcv.n_periods} periods")
    
    # 创建因子
    factor = VolAmplitudeImbalance(window=20, top_pct=0.2)
    print(f"\nFactor: {factor.name}")
    
    # 测试因子计算
    print("\nCalculating factor...")
    factor_data = factor.calculate(ohlcv)
    print(f"Factor shape: {factor_data.shape}")
    print(f"Factor stats: mean={factor_data.values.mean().mean():.4f}, std={factor_data.values.std().mean():.4f}")
    print(f"Factor range: [{factor_data.values.min().min():.4f}, {factor_data.values.max().max():.4f}]")
    
    # 数据泄露检测
    print("\n" + "="*60)
    print("Leakage Detection")
    print("="*60)
    detector = LeakageDetector(split_ratio=0.5, verbose=False)
    report = detector.detect(factor, ohlcv)
    
    print(f"Has leakage: {report.has_leakage}")
    print(f"Mismatch ratio: {report.mismatch_ratio:.6f}")
    print(f"Max absolute diff: {report.max_absolute_diff:.8f}")
    
    if report.has_leakage:
        print("\n[FAIL] Leakage detected! Please fix the factor implementation.")
        return False
    else:
        print("\n[OK] No leakage detected. Factor is ready for evaluation.")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
