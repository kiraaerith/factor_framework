"""
查询LowPriceRelativeVolume因子测评结果汇总
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage import DatabaseStorage

# 连接数据库
db = DatabaseStorage(r"E:\code_project\factor_eval_result\factor_eval.db")

# 查询因子结果
df = db.query_by_factor_type('LowPriceRelativeVolume')

print("="*80)
print("LowPriceRelativeVolume 因子测评结果汇总")
print("="*80)
print(f"总记录数: {len(df)}")

# 显示关键指标
if len(df) > 0:
    # 提取参数
    df['window'] = df['expression_params'].apply(lambda x: eval(x)['window'] if isinstance(x, str) else x['window'])
    df['top_pct'] = df['expression_params'].apply(lambda x: eval(x)['top_pct'] if isinstance(x, str) else x['top_pct'])
    df['rebalance_freq'] = df['evaluation_config'].apply(lambda x: eval(x)['rebalance_freq'] if isinstance(x, str) else x['rebalance_freq'])
    
    # 显示所有结果
    display_cols = ['expression_name', 'window', 'top_pct', 'rebalance_freq', 'rank_ic', 'rank_icir', 'sharpe', 'calmar', 'max_drawdown']
    print("\n所有测评结果:")
    print(df[display_cols].to_string(index=False))
    
    # 最佳Sharpe
    print("\n" + "="*80)
    print("最佳Sharpe比率 (Top 5)")
    print("="*80)
    best_sharpe = df.nlargest(5, 'sharpe')[['expression_name', 'window', 'top_pct', 'rebalance_freq', 'sharpe', 'rank_ic', 'calmar']]
    print(best_sharpe.to_string(index=False))
    
    # 最佳Rank IC
    print("\n" + "="*80)
    print("最佳Rank IC (Top 5)")
    print("="*80)
    best_ic = df.nlargest(5, 'rank_ic')[['expression_name', 'window', 'top_pct', 'rebalance_freq', 'rank_ic', 'rank_icir', 'sharpe']]
    print(best_ic.to_string(index=False))
    
    # 按调仓频率分组统计
    print("\n" + "="*80)
    print("按调仓频率分组统计")
    print("="*80)
    for rf in [5, 10, 20]:
        rf_df = df[df['rebalance_freq'] == rf]
        if len(rf_df) > 0:
            print(f"\nrebalance_freq={rf}:")
            print(f"  记录数: {len(rf_df)}")
            print(f"  Sharpe均值: {rf_df['sharpe'].mean():.4f}")
            print(f"  Sharpe最大: {rf_df['sharpe'].max():.4f}")
            print(f"  Rank IC均值: {rf_df['rank_ic'].mean():.4f}")
            print(f"  Rank IC最大: {rf_df['rank_ic'].max():.4f}")

print("\n" + "="*80)
print("查询完成!")
print("="*80)
