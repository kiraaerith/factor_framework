"""
查询PVCorr因子测评结果
"""

import sqlite3

def main():
    db_path = r'E:\code_project\factor_eval_result\factor_eval.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询最新记录
    print("="*80)
    print("最新15条记录")
    print("="*80)
    cursor.execute('''
        SELECT id, expression_name, factor_type, sharpe, rank_ic, calmar 
        FROM factor_evaluation_results 
        ORDER BY id DESC 
        LIMIT 15
    ''')
    rows = cursor.fetchall()
    print(f'{"ID":<5} | {"Expression Name":<35} | {"Factor Type":<25} | {"Sharpe":<8} | {"Rank IC":<8} | {"Calmar":<8}')
    print("-" * 120)
    for row in rows:
        print(f'{row[0]:<5} | {row[1]:<35} | {row[2]:<25} | {row[3]:>8.4f} | {row[4]:>8.4f} | {row[5]:>8.4f}')
    
    # 查询PVCorr相关记录
    print("\n" + "="*80)
    print("PriceVolumeCorrelation 因子记录")
    print("="*80)
    cursor.execute('''
        SELECT expression_name, sharpe, rank_ic, calmar 
        FROM factor_evaluation_results 
        WHERE expression_name LIKE '%PriceVolumeCorrelation%'
        ORDER BY expression_name
    ''')
    rows = cursor.fetchall()
    if rows:
        print(f'{"Expression Name":<40} | {"Sharpe":<8} | {"Rank IC":<8} | {"Calmar":<8}')
        print("-" * 80)
        for row in rows:
            print(f'{row[0]:<40} | {row[1]:>8.4f} | {row[2]:>8.4f} | {row[3]:>8.4f}')
        
        # 计算最佳参数
        cursor.execute('''
            SELECT expression_name, sharpe, rank_ic
            FROM factor_evaluation_results 
            WHERE expression_name LIKE '%PriceVolumeCorrelation%'
            ORDER BY sharpe DESC
            LIMIT 1
        ''')
        best_sharpe = cursor.fetchone()
        print(f"\n最佳Sharpe: {best_sharpe[0]} (Sharpe={best_sharpe[1]:.4f}, Rank IC={best_sharpe[2]:.4f})")
        
        cursor.execute('''
            SELECT expression_name, sharpe, rank_ic
            FROM factor_evaluation_results 
            WHERE expression_name LIKE '%PriceVolumeCorrelation%'
            ORDER BY rank_ic DESC
            LIMIT 1
        ''')
        best_rank_ic = cursor.fetchone()
        print(f"最佳Rank IC: {best_rank_ic[0]} (Sharpe={best_rank_ic[1]:.4f}, Rank IC={best_rank_ic[2]:.4f})")
    else:
        print("未找到PriceVolumeCorrelation相关记录")
    
    # 统计信息
    print("\n" + "="*80)
    print("数据库统计")
    print("="*80)
    cursor.execute('SELECT COUNT(*) FROM factor_evaluation_results')
    total = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT expression_name) FROM factor_evaluation_results')
    distinct_expr = cursor.fetchone()[0]
    print(f"总记录数: {total}")
    print(f"不同表达式数: {distinct_expr}")
    
    conn.close()

if __name__ == '__main__':
    main()
