"""
查询LowPriceRelativeVolume因子测评结果
"""

import sys
import os
import pandas as pd
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

# 显示所有列名
print("\n可用列名:")
print(df.columns.tolist())

# 显示前5行
print("\n前10行数据:")
with pd.option_context('display.max_columns', None, 'display.width', None):
    print(df.head(10))

print("\n" + "="*60)
print("查询完成!")
print("="*60)
