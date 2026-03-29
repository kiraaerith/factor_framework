"""
查询VolAmplitudeImbalance因子测评结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage import DatabaseStorage
import pandas as pd

# 连接数据库
db = DatabaseStorage(r"E:\code_project\factor_eval_result\factor_eval.db")

# 查看统计信息
print("="*80)
print("VolAmplitudeImbalance 因子测评结果")
print("="*80)

# 查询因子结果
df = db.query_by_factor_type('VolAmplitudeImbalance')
print(f"\n记录数: {len(df)}")

# 显示关键指标
if len(df) > 0:
    display_cols = [
        'expression_name', 'dataset_name', 'rank_ic_mean', 
        'sharpe_ratio', 'calmar_ratio', 'max_drawdown', 'annualized_return'
    ]
    print("\n关键指标:")
    print(df[display_cols].to_string(index=False))
    
    # 查询最佳参数组合
    print("\n" + "="*80)
    print("最佳参数组合 (按 Sharpe 排序)")
    print("="*80)
    best = df.nlargest(10, 'sharpe_ratio')[['expression_name', 'sharpe_ratio', 'rank_ic_mean', 'calmar_ratio']]
    print(best.to_string(index=False))
    
    # 按参数分组分析
    print("\n" + "="*80)
    print("测评结果汇总表")
    print("="*80)
    
    # 解析expression_name获取参数
    results = []
    for _, row in df.iterrows():
        name = row['expression_name']
        parts = name.split('_')
        window = None
        top_pct = None
        for part in parts:
            if part.startswith('w'):
                window = int(part[1:])
            elif part.startswith('p'):
                top_pct = float(part[1:])
        
        # 从dataset_name或rebalance_freq获取rf
        rf = 5  # 默认
        if 'rebalance_freq=10' in str(row.get('forward_period', '')):
            rf = 10
        elif 'rebalance_freq=20' in str(row.get('forward_period', '')):
            rf = 20
        
        results.append({
            'expression_name': name,
            'window': window,
            'top_pct': top_pct,
            'rank_ic_mean': row['rank_ic_mean'],
            'sharpe_ratio': row['sharpe_ratio'],
            'calmar_ratio': row['calmar_ratio'],
            'annualized_return': row['annualized_return'],
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 找出最佳Sharpe
    best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
    print("\n" + "="*80)
    print("最佳Sharpe参数组合:")
    print(f"  表达式: {best_sharpe['expression_name']}")
    print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.4f}")
    print(f"  Rank IC: {best_sharpe['rank_ic_mean']:.4f}")
    print(f"  Calmar: {best_sharpe['calmar_ratio']:.4f}")
    print(f"  年化收益: {best_sharpe['annualized_return']:.4f}")
    print(f"  最大回撤: {best_sharpe['max_drawdown']:.4f}")
    
    # 找出最佳Rank IC
    best_ic = df.loc[df['rank_ic_mean'].abs().idxmax()]
    print("\n最佳Rank IC参数组合:")
    print(f"  表达式: {best_ic['expression_name']}")
    print(f"  Rank IC: {best_ic['rank_ic_mean']:.4f}")
    print(f"  Sharpe: {best_ic['sharpe_ratio']:.4f}")

print("\n" + "="*80)
