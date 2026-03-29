"""
查询HighVolReturnSum因子测评结果

使用方法:
    python scripts/query/query_high_vol_return_sum.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from storage import DatabaseStorage


def main():
    # 连接数据库
    db_path = r"E:\code_project\factor_eval_result\factor_eval.db"
    db = DatabaseStorage(db_path)
    
    print("="*80)
    print("HighVolReturnSum 因子测评结果查询")
    print("="*80)
    
    # 查看统计信息
    print("\n" + "-"*80)
    print("数据库统计信息")
    print("-"*80)
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 查询HighVolReturnSum因子结果
    print("\n" + "-"*80)
    print("HighVolReturnSum 因子测评结果")
    print("-"*80)
    
    df = db.query_by_factor_type('HighVolReturnSum')
    print(f"记录数: {len(df)}")
    
    if len(df) > 0:
        # 显示关键指标
        display_cols = [
            'expression_name', 'dataset_name', 'rank_ic_mean', 
            'sharpe_ratio', 'calmar_ratio', 'max_drawdown', 'annualized_return'
        ]
        
        # 过滤存在的列
        available_cols = [c for c in display_cols if c in df.columns]
        print("\n关键指标:")
        print(df[available_cols].to_string(index=False))
        
        # 按Sharpe排序找最佳参数
        print("\n" + "-"*80)
        print("最佳参数组合 (按 Sharpe 排序 Top 10)")
        print("-"*80)
        
        if 'sharpe_ratio' in df.columns:
            best = df.nlargest(10, 'sharpe_ratio')[
                ['expression_name', 'sharpe_ratio', 'rank_ic_mean', 'calmar_ratio', 'annualized_return']
            ]
            print(best.to_string(index=False))
        
        # 按Rank IC排序找最佳参数
        print("\n" + "-"*80)
        print("最佳参数组合 (按 Rank IC 排序 Top 10)")
        print("-"*80)
        
        if 'rank_ic_mean' in df.columns:
            best_ic = df.nlargest(10, 'rank_ic_mean')[
                ['expression_name', 'rank_ic_mean', 'sharpe_ratio', 'calmar_ratio', 'annualized_return']
            ]
            print(best_ic.to_string(index=False))
        
        # 按调仓频率分组统计
        print("\n" + "-"*80)
        print("按调仓频率分组统计")
        print("-"*80)
        
        for rf in [5, 10, 20]:
            rf_df = df[df['expression_name'].str.contains(f'_rf{rf}_', na=False)]
            if len(rf_df) > 0:
                print(f"\n调仓频率 {rf}日:")
                print(f"  记录数: {len(rf_df)}")
                print(f"  平均Sharpe: {rf_df['sharpe_ratio'].mean():.4f}")
                print(f"  平均Rank IC: {rf_df['rank_ic_mean'].mean():.4f}")
                print(f"  最佳Sharpe: {rf_df['sharpe_ratio'].max():.4f}")
                if 'rebalance_freq' in rf_df.columns:
                    best_rf = rf_df.loc[rf_df['sharpe_ratio'].idxmax()]
                    print(f"  最佳参数: {best_rf['expression_name']}")
    
    print("\n" + "="*80)
    print("查询完成")
    print("="*80)


if __name__ == '__main__':
    main()
