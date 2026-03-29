"""
查询 HighVolAmplitude 因子测评结果
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

# 查询 HighVolAmplitude 因子结果
print("\n" + "="*60)
print("HighVolAmplitude 因子测评结果")
print("="*60)

df = db.query_by_factor_type('HighVolAmplitude')
print(f"记录数: {len(df)}")

# 显示关键指标
if len(df) > 0:
    print("\n可用列:", df.columns.tolist())
    
    # 尝试找到正确的列名
    sharpe_col = 'sharpe' if 'sharpe' in df.columns else 'sharpe_ratio'
    rank_ic_col = 'rank_ic' if 'rank_ic' in df.columns else 'rank_ic_mean'
    calmar_col = 'calmar' if 'calmar' in df.columns else 'calmar_ratio'
    
    display_cols = ['expression_name', 'dataset_name', sharpe_col, rank_ic_col, calmar_col]
    print("\n关键指标:")
    print(df[display_cols].to_string(index=False))

    # 查询最佳参数组合
    print("\n" + "="*60)
    print("最佳参数组合 (按 Sharpe 排序)")
    print("="*60)
    best = df.nlargest(5, sharpe_col)[['expression_name', sharpe_col, rank_ic_col, calmar_col]]
    print(best.to_string(index=False))
else:
    print("未找到 HighVolAmplitude 因子的测评记录")
