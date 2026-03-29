"""
分析不同调仓频率(rebalance_freq)下动量因子的表现
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import pandas as pd
import json

DB_PATH = r"E:\code_project\factor_eval_result\factor_eval.db"

def extract_rebalance_freq(config_json):
    """从测评配置中提取调仓频率"""
    try:
        config = json.loads(config_json)
        return config.get('rebalance_freq', 'unknown')
    except:
        return 'unknown'

def main():
    print("="*80)
    print("动量因子在不同调仓频率下的表现分析")
    print(f"数据库路径: {DB_PATH}")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 读取动量因子数据
    df = pd.read_sql_query(
        "SELECT * FROM factor_evaluation_results WHERE factor_type = 'MomentumFactor'", 
        conn
    )
    
    # 提取调仓频率
    df['rebalance_freq'] = df['evaluation_config'].apply(extract_rebalance_freq)
    
    # 提取offset和lookback
    df['offset'] = df['expression_name'].str.extract(r'o(\d+)_l').astype(int)
    df['lookback'] = df['expression_name'].str.extract(r'l(\d+)').astype(int)
    
    print(f"\n动量因子总记录数: {len(df)}")
    print(f"\n按调仓频率分布:")
    rf_dist = df['rebalance_freq'].value_counts().sort_index()
    print(rf_dist.to_string())
    
    # 1. 按调仓频率的整体统计
    print("\n" + "="*80)
    print("一、按调仓频率的整体表现对比")
    print("="*80)
    
    rf_stats = df.groupby('rebalance_freq').agg({
        'sharpe': ['mean', 'std', 'max', 'min'],
        'rank_ic': ['mean', 'std', 'max'],
        'rank_icir': ['mean', 'std'],
        'calmar': ['mean', 'std'],
        'max_drawdown': ['mean', 'std']
    }).round(4)
    
    rf_stats.columns = [
        'Sharpe均值', 'Sharpe标准差', 'Sharpe最大', 'Sharpe最小',
        'RankIC均值', 'RankIC标准差', 'RankIC最大',
        'RankICIR均值', 'RankICIR标准差',
        'Calmar均值', 'Calmar标准差',
        '最大回撤均值', '最大回撤标准差'
    ]
    print(rf_stats.to_string())
    
    # 2. 各回看周期在不同调仓频率下的表现
    print("\n" + "="*80)
    print("二、各回看周期(Y)在不同调仓频率下的Sharpe表现")
    print("="*80)
    
    pivot_y = df.pivot_table(
        values='sharpe', 
        index='lookback', 
        columns='rebalance_freq', 
        aggfunc='mean'
    ).round(4)
    print("\nSharpe均值:")
    print(pivot_y.to_string())
    
    print("\n" + "-"*80)
    pivot_y_ic = df.pivot_table(
        values='rank_ic', 
        index='lookback', 
        columns='rebalance_freq', 
        aggfunc='mean'
    ).round(4)
    print("\nRank IC均值:")
    print(pivot_y_ic.to_string())
    
    print("\n" + "-"*80)
    pivot_y_icir = df.pivot_table(
        values='rank_icir', 
        index='lookback', 
        columns='rebalance_freq', 
        aggfunc='mean'
    ).round(4)
    print("\nRank ICIR均值:")
    print(pivot_y_icir.to_string())
    
    # 3. 各延迟周期在不同调仓频率下的表现
    print("\n" + "="*80)
    print("三、各延迟周期(X)在不同调仓频率下的Sharpe表现")
    print("="*80)
    
    pivot_x = df.pivot_table(
        values='sharpe', 
        index='offset', 
        columns='rebalance_freq', 
        aggfunc='mean'
    ).round(4)
    print("\nSharpe均值:")
    print(pivot_x.to_string())
    
    print("\n" + "-"*80)
    pivot_x_ic = df.pivot_table(
        values='rank_ic', 
        index='offset', 
        columns='rebalance_freq', 
        aggfunc='mean'
    ).round(4)
    print("\nRank IC均值:")
    print(pivot_x_ic.to_string())
    
    # 4. 每个调仓频率下的Top 5因子
    print("\n" + "="*80)
    print("四、各调仓频率下表现最佳的Top 5因子")
    print("="*80)
    
    for rf in sorted(df['rebalance_freq'].unique()):
        if rf == 'unknown':
            continue
        print(f"\n【调仓频率 = {rf}日】")
        print("-"*60)
        rf_df = df[df['rebalance_freq'] == rf]
        top5 = rf_df.nlargest(5, 'sharpe')[['expression_name', 'offset', 'lookback', 
                                            'sharpe', 'rank_ic', 'rank_icir', 'calmar']]
        print(top5.to_string(index=False))
    
    # 5. 不同调仓频率下的最佳参数组合对比
    print("\n" + "="*80)
    print("五、不同调仓频率下的最佳参数组合")
    print("="*80)
    
    for rf in sorted(df['rebalance_freq'].unique()):
        if rf == 'unknown':
            continue
        print(f"\n【调仓频率 = {rf}日】")
        print("-"*60)
        rf_df = df[df['rebalance_freq'] == rf]
        
        # 按Sharpe排序
        best_sharpe = rf_df.loc[rf_df['sharpe'].idxmax()]
        print(f"最佳Sharpe: {best_sharpe['expression_name']}")
        print(f"  Sharpe={best_sharpe['sharpe']:.4f}, RankIC={best_sharpe['rank_ic']:.4f}, Calmar={best_sharpe['calmar']:.4f}")
        
        # 按Rank IC排序
        best_rank_ic = rf_df.loc[rf_df['rank_ic'].idxmax()]
        print(f"\n最佳Rank IC: {best_rank_ic['expression_name']}")
        print(f"  RankIC={best_rank_ic['rank_ic']:.4f}, RankICIR={best_rank_ic['rank_icir']:.4f}, Sharpe={best_rank_ic['sharpe']:.4f}")
        
        # 按Rank ICIR排序
        best_rank_icir = rf_df.loc[rf_df['rank_icir'].idxmax()]
        print(f"\n最佳Rank ICIR: {best_rank_icir['expression_name']}")
        print(f"  RankICIR={best_rank_icir['rank_icir']:.4f}, RankIC={best_rank_icir['rank_ic']:.4f}, Sharpe={best_rank_icir['sharpe']:.4f}")
    
    # 6. 统计显著性分析
    print("\n" + "="*80)
    print("六、不同调仓频率下各回看周期的样本数量")
    print("="*80)
    
    count_table = df.pivot_table(
        values='id', 
        index='lookback', 
        columns='rebalance_freq', 
        aggfunc='count',
        fill_value=0
    )
    print(count_table.to_string())
    
    conn.close()
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == '__main__':
    main()
