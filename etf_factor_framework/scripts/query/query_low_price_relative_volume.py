"""
查询LowPriceRelativeVolume因子测评结果
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
print("LowPriceRelativeVolume 因子测评结果")
print("="*60)

# 按因子类型查询
df = db.query_by_factor_type('LowPriceRelativeVolume')
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
    best = df.nlargest(5, 'sharpe_ratio')[['expression_name', 'sharpe_ratio', 'rank_ic_mean', 'calmar_ratio']]
    print(best.to_string(index=False))
    
    # 最佳Rank IC
    print("\n" + "="*60)
    print("最佳Rank IC (按 Rank IC 排序)")
    print("="*60)
    best_ic = df.nlargest(5, 'rank_ic_mean')[['expression_name', 'sharpe_ratio', 'rank_ic_mean', 'calmar_ratio']]
    print(best_ic.to_string(index=False))

print("\n" + "="*60)
print("查询完成!")
print("="*60)
