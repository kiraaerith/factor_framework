"""
查询 factor_eval.db 数据库汇总信息
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import pandas as pd
import json

DB_PATH = r"E:\code_project\factor_eval_result\factor_eval.db"

def main():
    print("="*80)
    print("Factor Evaluation Database 汇总信息")
    print(f"数据库路径: {DB_PATH}")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 基础统计
    df = pd.read_sql_query("SELECT * FROM factor_evaluation_results", conn)
    print(f"\n总记录数: {len(df)}")
    
    # 因子类型统计
    print("\n" + "-"*80)
    print("按因子类型统计")
    print("-"*80)
    factor_stats = df.groupby('factor_type').agg({
        'id': 'count',
        'sharpe': ['mean', 'max', 'min'],
        'rank_ic': ['mean', 'max'],
        'rank_icir': 'mean'
    }).round(4)
    factor_stats.columns = ['记录数', 'Sharpe均值', 'Sharpe最大', 'Sharpe最小', 'RankIC均值', 'RankIC最大', 'RankICIR均值']
    print(factor_stats.to_string())
    
    # 数据集统计
    print("\n" + "-"*80)
    print("按数据集统计")
    print("-"*80)
    dataset_stats = df.groupby('dataset_name').size()
    print(dataset_stats.to_string())
    
    # 最佳Sharpe前20
    print("\n" + "-"*80)
    print("Sharpe比率 Top 20")
    print("-"*80)
    top_sharpe = df.nlargest(20, 'sharpe')[['expression_name', 'factor_type', 'sharpe', 'rank_ic', 'rank_icir', 'calmar']]
    print(top_sharpe.to_string(index=False))
    
    # 最佳Rank IC前20
    print("\n" + "-"*80)
    print("Rank IC Top 20")
    print("-"*80)
    top_rank_ic = df.nlargest(20, 'rank_ic')[['expression_name', 'factor_type', 'rank_ic', 'rank_icir', 'sharpe']]
    print(top_rank_ic.to_string(index=False))
    
    # 最佳Rank ICIR前20
    print("\n" + "-"*80)
    print("Rank ICIR Top 20")
    print("-"*80)
    top_rank_icir = df.nlargest(20, 'rank_icir')[['expression_name', 'factor_type', 'rank_icir', 'rank_ic', 'sharpe']]
    print(top_rank_icir.to_string(index=False))
    
    # 动量因子专项分析
    print("\n" + "="*80)
    print("动量因子 (MomentumFactor) 专项分析")
    print("="*80)
    momentum_df = df[df['factor_type'] == 'MomentumFactor'].copy()
    print(f"动量因子记录数: {len(momentum_df)}")
    
    if len(momentum_df) > 0:
        # 解析参数
        momentum_df['offset'] = momentum_df['expression_name'].str.extract(r'o(\d+)_l').astype(int)
        momentum_df['lookback'] = momentum_df['expression_name'].str.extract(r'l(\d+)').astype(int)
        
        # 按回看周期统计
        print("\n按回看周期(Y)统计:")
        y_stats = momentum_df.groupby('lookback').agg({
            'sharpe': 'mean',
            'rank_ic': 'mean',
            'rank_icir': 'mean',
            'calmar': 'mean'
        }).round(4)
        print(y_stats.to_string())
        
        # 按延迟周期统计
        print("\n按延迟周期(X)统计:")
        x_stats = momentum_df.groupby('offset').agg({
            'sharpe': 'mean',
            'rank_ic': 'mean',
            'rank_icir': 'mean',
            'calmar': 'mean'
        }).round(4)
        print(x_stats.to_string())
        
        # 最佳组合
        print("\n动量因子 Sharpe Top 10:")
        top_momentum = momentum_df.nlargest(10, 'sharpe')[['expression_name', 'offset', 'lookback', 'sharpe', 'rank_ic', 'rank_icir']]
        print(top_momentum.to_string(index=False))
    
    conn.close()
    
    print("\n" + "="*80)
    print("查询完成！")
    print("="*80)

if __name__ == '__main__':
    main()
