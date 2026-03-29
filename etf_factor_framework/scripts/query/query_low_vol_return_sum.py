"""
查询LowVolReturnSum因子测评结果
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from storage import DatabaseStorage


def main():
    db_path = r"E:\code_project\factor_eval_result\factor_eval.db"
    db = DatabaseStorage(db_path)
    
    print("="*80)
    print("LowVolReturnSum 因子测评结果查询")
    print("="*80)
    
    df = db.query_by_factor_type('LowVolReturnSum')
    print(f"记录数: {len(df)}")
    
    if len(df) > 0:
        display_cols = [
            'expression_name', 'dataset_name', 'rank_ic_mean', 
            'sharpe_ratio', 'calmar_ratio', 'max_drawdown', 'annualized_return'
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        print("\n关键指标:")
        print(df[available_cols].to_string(index=False))
        
        print("\n" + "-"*80)
        print("最佳参数组合 (按 Sharpe 排序 Top 10)")
        print("-"*80)
        
        if 'sharpe_ratio' in df.columns:
            best = df.nlargest(10, 'sharpe_ratio')[
                ['expression_name', 'sharpe_ratio', 'rank_ic_mean', 'calmar_ratio', 'annualized_return']
            ]
            print(best.to_string(index=False))
        
        print("\n" + "-"*80)
        print("最佳参数组合 (按 Rank IC 排序 Top 10)")
        print("-"*80)
        
        if 'rank_ic_mean' in df.columns:
            best_ic = df.nlargest(10, 'rank_ic_mean')[
                ['expression_name', 'rank_ic_mean', 'sharpe_ratio', 'calmar_ratio', 'annualized_return']
            ]
            print(best_ic.to_string(index=False))
    
    print("\n" + "="*80)
    print("查询完成")
    print("="*80)


if __name__ == '__main__':
    main()
