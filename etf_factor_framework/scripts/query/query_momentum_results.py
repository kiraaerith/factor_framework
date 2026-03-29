"""
查询动量因子测评结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage import DatabaseStorage
import pandas as pd

# 连接数据库
db = DatabaseStorage(r"E:\code_project\factor_eval_result\factor_eval.db")

# 查询动量因子结果
print("="*80)
print("动量因子测评结果汇总")
print("="*80)

df = db.query_by_factor_type('MomentumFactor')
print(f"总记录数: {len(df)}")

# 按回看周期分组统计
if len(df) > 0:
    # 解析offset和lookback
    df['offset'] = df['expression_name'].str.extract(r'o(\d+)_l').astype(int)
    df['lookback'] = df['expression_name'].str.extract(r'l(\d+)').astype(int)
    
    print("\n" + "="*80)
    print("按回看周期分组的最佳因子（按Sharpe排序）")
    print("="*80)
    
    for y in [2, 5, 8, 13, 21, 34, 55, 89, 144]:
        y_df = df[df['lookback'] == y]
        if len(y_df) > 0:
            best = y_df.loc[y_df['sharpe'].idxmax()]
            print(f"Y={y:3d}: {best['expression_name']:20s} | "
                  f"Sharpe={best['sharpe']:7.4f} | "
                  f"Rank IC={best['rank_ic']:7.4f} | "
                  f"ICIR={best['rank_icir']:7.4f}")
    
    # 显示关键指标（按Sharpe排序前20）
    print("\n" + "="*80)
    print("所有因子按Sharpe排序（Top 20）")
    print("="*80)
    display_cols = [
        'expression_name', 'rank_ic', 'rank_icir', 
        'sharpe', 'calmar', 'max_drawdown', 'ic'
    ]
    top20 = df.nlargest(20, 'sharpe')[display_cols]
    print(top20.to_string(index=False))
    
    # 统计各回看周期的平均表现
    print("\n" + "="*80)
    print("各回看周期平均表现")
    print("="*80)
    
    summary = df.groupby('lookback').agg({
        'rank_ic': 'mean',
        'rank_icir': 'mean',
        'sharpe': 'mean',
        'calmar': 'mean',
        'ic': 'mean'
    }).round(4)
    print(summary.to_string())
    
    # 统计各延迟周期的平均表现
    print("\n" + "="*80)
    print("各延迟周期平均表现")
    print("="*80)
    
    summary_offset = df.groupby('offset').agg({
        'rank_ic': 'mean',
        'rank_icir': 'mean',
        'sharpe': 'mean',
        'calmar': 'mean',
        'ic': 'mean'
    }).round(4)
    print(summary_offset.to_string())

print("\n" + "="*80)
print("测评完成！")
print("="*80)
