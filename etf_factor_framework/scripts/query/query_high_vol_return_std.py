"""
查询 HighVolReturnStd 因子测评结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage import DatabaseStorage
import pandas as pd

# 连接数据库
db = DatabaseStorage(r"E:\code_project\factor_eval_result\factor_eval.db")

# 查看统计信息
print("="*60)
print("数据库统计信息")
print("="*60)
stats = db.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

# 查询 HighVolReturnStd 因子结果
print("\n" + "="*60)
print("HighVolReturnStd 因子测评结果")
print("="*60)

df = db.query_by_factor_type('HighVolReturnStd')
print(f"记录数: {len(df)}")
print(f"\n可用列: {list(df.columns)}")

# 显示关键指标
if len(df) > 0:
    # 查询最佳参数组合
    print("\n" + "="*60)
    print("所有结果 (按 Sharpe 排序)")
    print("="*60)
    
    # 解析参数
    def parse_params(name):
        parts = name.split('_')
        window = None
        top_pct = None
        for p in parts:
            if p.startswith('w') and p[1:].isdigit():
                window = int(p[1:])
            if p.startswith('p') and len(p) > 1:
                try:
                    top_pct = float(p[1:])
                except:
                    pass
        return window, top_pct
    
    df['window'] = df['expression_name'].apply(lambda x: parse_params(x)[0])
    df['top_pct'] = df['expression_name'].apply(lambda x: parse_params(x)[1])
    
    # 显示关键列
    key_cols = ['expression_name', 'window', 'top_pct', 'rebalance_freq', 'sharpe', 'calmar', 'rank_ic', 'rank_icir']
    available_cols = [c for c in key_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))
    
    # 最佳Sharpe
    print("\n" + "="*60)
    print("最佳参数组合 (按 Sharpe 排序 Top 5)")
    print("="*60)
    best = df.nlargest(5, 'sharpe')[available_cols]
    print(best.to_string(index=False))
    
    # 汇总表
    print("\n" + "="*60)
    print("Sharpe比率汇总表")
    print("="*60)
    pivot = df.pivot_table(
        index=['window', 'top_pct'],
        columns='rebalance_freq',
        values='sharpe',
        aggfunc='first'
    )
    print(pivot.to_string())
