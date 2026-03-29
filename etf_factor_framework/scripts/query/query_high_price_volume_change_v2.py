"""
查询 HighPriceVolumeChange 因子测评结果
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage import DatabaseStorage

# 连接数据库
db = DatabaseStorage(r"E:\code_project\factor_eval_result\factor_eval.db")

# 查询 HighPriceVolumeChange 因子结果
df = db.query_by_factor_type('HighPriceVolumeChange')
print('HighPriceVolumeChange Factor Evaluation Results:')
print('='*80)
print(df[['expression_name', 'sharpe', 'rank_ic', 'calmar', 'max_drawdown']].to_string(index=False))
print()
print('='*80)
print('Top 10 by Sharpe:')
best = df.nlargest(10, 'sharpe')[['expression_name', 'sharpe', 'rank_ic', 'calmar']]
print(best.to_string(index=False))
print()
print('='*80)
best_row = df.loc[df['sharpe'].idxmax()]
print(f'Best Sharpe: {best_row["sharpe"]:.4f}')
print(f'Expression: {best_row["expression_name"]}')
print(f'Rank IC: {best_row["rank_ic"]:.4f}')
print(f'Calmar: {best_row["calmar"]:.4f}')
