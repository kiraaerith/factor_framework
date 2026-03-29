"""
查询 LowVolChangeReturnSum 因子测评结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage import DatabaseStorage

# 连接数据库
db = DatabaseStorage(r"E:\code_project\factor_eval_result\factor_eval.db")

# 查看统计信息
print("="*60)
print("数据库统计信息")
print("="*60)
stats = db.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

# 查询因子结果
print("\n" + "="*60)
print("LowVolChangeReturnSum 因子测评结果")
print("="*60)

# 按因子类型查询
df = db.query_by_factor_type('LowVolChangeReturnSum')
print(f"记录数: {len(df)}")

# 显示关键指标
if len(df) > 0:
    display_cols = [
        'expression_name', 'dataset_name', 'rank_ic', 
        'sharpe', 'calmar', 'max_drawdown'
    ]
    print("\n关键指标:")
    print(df[display_cols].to_string(index=False))

# 查询最佳参数组合
print("\n" + "="*60)
print("最佳参数组合 (按 Sharpe 排序)")
print("="*60)
best = df.nlargest(10, 'sharpe')[['expression_name', 'sharpe', 'rank_ic', 'calmar']]
print(best.to_string(index=False))

# 提取参数信息并整理
print("\n" + "="*60)
print("详细超参组合分析")
print("="*60)

results = []
for _, row in df.iterrows():
    name = row['expression_name']
    # 解析参数
    import re
    match = re.search(r'w(\d+)_p([\d.]+)', name)
    if match:
        window = int(match.group(1))
        top_pct = float(match.group(2))
        rebalance_freq = row.get('rebalance_freq', 'unknown')
        results.append({
            'window': window,
            'top_pct': top_pct,
            'rebalance_freq': rebalance_freq,
            'sharpe': row['sharpe'],
            'rank_ic': row['rank_ic'],
            'calmar': row['calmar']
        })

import pandas as pd
results_df = pd.DataFrame(results)
if len(results_df) > 0:
    print(results_df.to_string(index=False))
    
    # 最佳Sharpe
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    print("\n" + "="*60)
    print("最佳Sharpe参数组合:")
    print(f"  window={best_sharpe['window']}, top_pct={best_sharpe['top_pct']}, rf={best_sharpe['rebalance_freq']}")
    print(f"  Sharpe={best_sharpe['sharpe']:.4f}, Rank IC={best_sharpe['rank_ic']:.4f}")
    
    # 最佳Rank IC
    best_rank_ic = results_df.loc[results_df['rank_ic'].idxmax()]
    print("\n最佳Rank IC参数组合:")
    print(f"  window={int(best_rank_ic['window'])}, top_pct={best_rank_ic['top_pct']}, rf={best_rank_ic['rebalance_freq']}")
    print(f"  Sharpe={best_rank_ic['sharpe']:.4f}, Rank IC={best_rank_ic['rank_ic']:.4f}")
