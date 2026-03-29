"""
查询数据库中的评估结果
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from storage import DatabaseStorage

# 连接到数据库
db_path = r"E:\code_project\factor_eval_result\factor_eval.db"
db = DatabaseStorage(db_path)

print("="*70)
print("数据库查询结果")
print("="*70)

# 获取统计信息
print("\n【数据库统计】")
stats = db.get_statistics()
print(f"  总记录数: {stats['total_records']}")
print(f"  不同表达式: {stats['distinct_expressions']}")
print(f"  不同数据集: {stats['distinct_datasets']}")

# 所有表达式
print("\n【所有表达式名称】")
expressions = db.get_distinct_expressions()
for expr in expressions:
    print(f"  - {expr}")

# 查询所有记录
print("\n【所有评估记录】")
df = db.execute_custom_query("""
    SELECT expression_name, dataset_name, 
           sharpe, max_drawdown, calmar, 
           ic, icir, rank_ic, rank_icir,
           created_at
    FROM factor_evaluation_results
    ORDER BY created_at DESC
""")

# 美化输出
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print(df.to_string(index=False))

# 按夏普比率排序
print("\n【按夏普比率排序 (Top 5)】")
df_sorted = df.sort_values('sharpe', ascending=False).head(5)
print(df_sorted[['expression_name', 'sharpe', 'calmar', 'rank_ic', 'rank_icir']].to_string(index=False))

print("\n" + "="*70)
