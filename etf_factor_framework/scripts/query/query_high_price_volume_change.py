"""
查询 HighPriceVolumeChange 因子测评结果
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

# 查询 HighPriceVolumeChange 因子结果
print("\n" + "="*60)
print("HighPriceVolumeChange 因子测评结果")
print("="*60)

# 按因子类型查询
df = db.query_by_factor_type('HighPriceVolumeChange')
print(f"记录数: {len(df)}")

# 显示关键指标
if len(df) > 0:
    display_cols = [
        'expression_name', 'dataset_name', 'rank_ic_mean', 
        'sharpe_ratio', 'calmar_ratio', 'max_drawdown', 'annualized_return'
    ]
    print("\n关键指标:")
    print(df[display_cols].to_string(index=False))
    
    # 查询最佳参数组合
    print("\n" + "="*60)
    print("最佳参数组合 (按 Sharpe 排序)")
    print("="*60)
    best = df.nlargest(10, 'sharpe_ratio')[['expression_name', 'sharpe_ratio', 'rank_ic_mean', 'calmar_ratio']]
    print(best.to_string(index=False))
    
    # 显示最佳结果的详细信息
    print("\n" + "="*60)
    print("最佳 Sharpe 的详细信息")
    print("="*60)
    best_row = df.loc[df['sharpe_ratio'].idxmax()]
    print(f"表达式名称: {best_row['expression_name']}")
    print(f"Sharpe比率: {best_row['sharpe_ratio']:.4f}")
    print(f"Rank IC: {best_row['rank_ic_mean']:.4f}")
    print(f"Calmar比率: {best_row['calmar_ratio']:.4f}")
    print(f"年化收益率: {best_row['annualized_return']:.4f}")
    print(f"最大回撤: {best_row['max_drawdown']:.4f}")

print("\n" + "="*60)
print("测评完成!")
print("="*60)
