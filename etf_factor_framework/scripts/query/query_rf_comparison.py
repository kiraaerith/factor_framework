"""
Compare momentum factor performance across different rebalance frequencies
"""

import sqlite3
import pandas as pd
import json

DB_PATH = r"E:\code_project\factor_eval_result\factor_eval.db"
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query(
    "SELECT * FROM factor_evaluation_results WHERE factor_type = 'MomentumFactor'", 
    conn
)

df['rebalance_freq'] = df['evaluation_config'].apply(lambda x: json.loads(x).get('rebalance_freq', 'unknown'))
df['offset'] = df['expression_name'].str.extract(r'o(\d+)_l').astype(int)
df['lookback'] = df['expression_name'].str.extract(r'l(\d+)').astype(int)

print("="*80)
print("MOMENTUM FACTOR PERFORMANCE ACROSS REBALANCE FREQUENCIES")
print("="*80)

print("\n1. OVERALL STATISTICS BY REBALANCE FREQUENCY")
print("-"*80)
rf_stats = df.groupby('rebalance_freq').agg({
    'sharpe': ['mean', 'std', 'max', 'min'],
    'rank_ic': ['mean', 'std', 'max'],
    'rank_icir': ['mean', 'std'],
    'calmar': ['mean', 'std']
}).round(4)
print(rf_stats)

print("\n2. SHARPE RATIO BY LOOKBACK PERIOD AND REBALANCE FREQ")
print("-"*80)
pivot = df.pivot_table(values='sharpe', index='lookback', columns='rebalance_freq', aggfunc='mean').round(4)
print(pivot)

print("\n3. RANK IC BY LOOKBACK PERIOD AND REBALANCE FREQ")
print("-"*80)
pivot_ic = df.pivot_table(values='rank_ic', index='lookback', columns='rebalance_freq', aggfunc='mean').round(4)
print(pivot_ic)

print("\n4. RANK ICIR BY LOOKBACK PERIOD AND REBALANCE FREQ")
print("-"*80)
pivot_icir = df.pivot_table(values='rank_icir', index='lookback', columns='rebalance_freq', aggfunc='mean').round(4)
print(pivot_icir)

print("\n5. SHARPE BY OFFSET AND REBALANCE FREQ")
print("-"*80)
pivot_x = df.pivot_table(values='sharpe', index='offset', columns='rebalance_freq', aggfunc='mean').round(4)
print(pivot_x)

print("\n6. TOP 3 FACTORS BY REBALANCE FREQUENCY")
print("-"*80)
for rf in [5, 10, 20]:
    print(f"\nRebalance Freq = {rf} days:")
    rf_df = df[df['rebalance_freq'] == rf]
    top3 = rf_df.nlargest(3, 'sharpe')[['expression_name', 'offset', 'lookback', 'sharpe', 'rank_ic', 'rank_icir']]
    print(top3.to_string(index=False))

print("\n7. BEST FACTOR BY REBALANCE FREQUENCY")
print("-"*80)
for rf in [5, 10, 20]:
    rf_df = df[df['rebalance_freq'] == rf]
    best_sharpe = rf_df.loc[rf_df['sharpe'].idxmax()]
    best_rank_ic = rf_df.loc[rf_df['rank_ic'].idxmax()]
    best_rank_icir = rf_df.loc[rf_df['rank_icir'].idxmax()]
    
    print(f"\nRF = {rf} days:")
    print(f"  Best Sharpe:  {best_sharpe['expression_name']} (Sharpe={best_sharpe['sharpe']:.4f}, RankIC={best_sharpe['rank_ic']:.4f})")
    print(f"  Best Rank IC: {best_rank_ic['expression_name']} (RankIC={best_rank_ic['rank_ic']:.4f}, Sharpe={best_rank_ic['sharpe']:.4f})")
    print(f"  Best ICIR:    {best_rank_icir['expression_name']} (ICIR={best_rank_icir['rank_icir']:.4f}, RankIC={best_rank_icir['rank_ic']:.4f})")

conn.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
