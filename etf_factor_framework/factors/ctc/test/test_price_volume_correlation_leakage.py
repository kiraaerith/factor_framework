"""
PriceVolumeCorrelation 因子数据泄露检测脚本
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from core.ohlcv_data import OHLCVData
from factors.ctc.price_volume_correlation import PVCorr, DPVCorr
from factors.leakage_detector import detect_leakage


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
    
    # 测试PVCorr因子
    print("\n" + "="*60)
    print("数据泄露检测 - PVCorr (收盘价与成交量相关系数)")
    print("="*60)
    
    factor_pv = PVCorr(window=20)
    report_pv = detect_leakage(factor_pv, ohlcv, split_ratio=0.5, verbose=False)
    
    print("\n" + "="*60)
    print("检测结果总结 - PVCorr")
    print("="*60)
    print(f"  是否泄露: {report_pv.has_leakage}")
    print(f"  不匹配比例: {report_pv.mismatch_ratio:.4%}")
    print(f"  最大差异: {report_pv.max_absolute_diff:.8f}")
    if report_pv.has_leakage:
        print(f"[WARNING] {factor_pv.name} 存在未来数据泄露!")
        return False
    else:
        print(f"[OK] {factor_pv.name} 无数据泄露，可以继续测评流程")
    
    # 测试DPVCorr因子
    print("\n" + "="*60)
    print("数据泄露检测 - DPVCorr (收益率与成交量相关系数)")
    print("="*60)
    
    factor_dpv = DPVCorr(window=20)
    report_dpv = detect_leakage(factor_dpv, ohlcv, split_ratio=0.5, verbose=False)
    
    print("\n" + "="*60)
    print("检测结果总结 - DPVCorr")
    print("="*60)
    print(f"  是否泄露: {report_dpv.has_leakage}")
    print(f"  不匹配比例: {report_dpv.mismatch_ratio:.4%}")
    print(f"  最大差异: {report_dpv.max_absolute_diff:.8f}")
    if report_dpv.has_leakage:
        print(f"[WARNING] {factor_dpv.name} 存在未来数据泄露!")
        return False
    else:
        print(f"[OK] {factor_dpv.name} 无数据泄露，可以继续测评流程")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
