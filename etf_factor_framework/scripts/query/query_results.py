"""查询VolAmplitudeImbalance因子测评结果"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage import DatabaseStorage
import pandas as pd

db = DatabaseStorage(r"E:\code_project\factor_eval_result\factor_eval.db")
df = db.query_by_factor_type('VolAmplitudeImbalance')

print("="*80)
print("VolAmplitudeImbalance Factor Evaluation Results")
print("="*80)
print(f"Total records: {len(df)}")

def parse_params(name):
    parts = name.split('_')
    window = None
    top_pct = None
    for part in parts:
        if part.startswith('w'):
            window = int(part[1:])
        elif part.startswith('p'):
            top_pct = float(part[1:])
    return window, top_pct

df['window'] = df['expression_name'].apply(lambda x: parse_params(x)[0])
df['top_pct'] = df['expression_name'].apply(lambda x: parse_params(x)[1])

def get_rf(config):
    try:
        import json
        c = json.loads(config)
        return c.get('rebalance_freq', 5)
    except:
        return 5

df['rf'] = df['evaluation_config'].apply(get_rf)

print("\n" + "="*80)
print("Sharpe Ratio by Parameters")
print("="*80)
pivot = df.pivot_table(values='sharpe', index=['window', 'top_pct'], columns='rf', aggfunc='mean')
print(pivot.to_string())

print("\n" + "="*80)
print("Rank IC by Parameters")
print("="*80)
pivot_ic = df.pivot_table(values='rank_ic', index=['window', 'top_pct'], columns='rf', aggfunc='mean')
print(pivot_ic.to_string())

print("\n" + "="*80)
print("Top 10 Parameter Combinations (by Sharpe)")
print("="*80)
best = df.nlargest(10, 'sharpe')[['expression_name', 'window', 'top_pct', 'rf', 'sharpe', 'rank_ic', 'calmar']]
print(best.to_string(index=False))

best_sharpe = df.loc[df['sharpe'].idxmax()]
print("\n" + "="*80)
print("BEST PARAMETER COMBINATION:")
print(f"  Expression: {best_sharpe['expression_name']}")
print(f"  Window: {best_sharpe['window']}, top_pct: {best_sharpe['top_pct']}, rf: {best_sharpe['rf']}")
print(f"  Sharpe: {best_sharpe['sharpe']:.4f}")
print(f"  Rank IC: {best_sharpe['rank_ic']:.4f}")
print(f"  Calmar: {best_sharpe['calmar']:.4f}")
print("="*80)
