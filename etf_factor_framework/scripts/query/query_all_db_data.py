"""
查询 factor_eval.db 数据库中的所有数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import pandas as pd

DB_PATH = r"E:\code_project\factor_eval_result\factor_eval.db"

def get_tables(conn):
    """获取所有表名"""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]

def get_table_info(conn, table_name):
    """获取表结构信息"""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def get_table_count(conn, table_name):
    """获取表记录数"""
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]

def query_table(conn, table_name, limit=100):
    """查询表数据"""
    return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)

def main():
    print("="*80)
    print("Factor Evaluation Database 数据查询")
    print(f"数据库路径: {DB_PATH}")
    print("="*80)
    
    # 连接数据库
    conn = sqlite3.connect(DB_PATH)
    
    # 获取所有表
    tables = get_tables(conn)
    print(f"\n数据库包含 {len(tables)} 个表:")
    for i, table in enumerate(tables, 1):
        count = get_table_count(conn, table)
        print(f"  {i}. {table} ({count} 条记录)")
    
    # 逐个表查看数据
    for table_name in tables:
        print("\n" + "="*80)
        print(f"表: {table_name}")
        print("="*80)
        
        # 表结构
        print("\n【表结构】")
        columns = get_table_info(conn, table_name)
        for col in columns:
            cid, name, dtype, notnull, default_val, pk = col
            pk_str = " (PK)" if pk else ""
            null_str = " NOT NULL" if notnull else ""
            print(f"  {name}: {dtype}{pk_str}{null_str}")
        
        # 表数据
        count = get_table_count(conn, table_name)
        print(f"\n【表数据】共 {count} 条记录")
        
        if count > 0:
            df = query_table(conn, table_name, limit=50)
            print(f"\n显示前 {min(count, 50)} 条记录:")
            print(df.to_string(index=False))
            
            if count > 50:
                print(f"\n... 还有 {count - 50} 条记录未显示 ...")
    
    # 统计分析
    print("\n" + "="*80)
    print("统计分析")
    print("="*80)
    
    if 'factor_evaluations' in tables:
        print("\n【因子测评统计】")
        df = pd.read_sql_query("""
            SELECT 
                factor_type,
                COUNT(*) as count,
                AVG(sharpe) as avg_sharpe,
                AVG(rank_ic) as avg_rank_ic,
                AVG(rank_icir) as avg_rank_icir
            FROM factor_evaluations 
            GROUP BY factor_type
        """, conn)
        print(df.to_string(index=False))
        
        print("\n【按因子类型和回看周期统计】")
        df = pd.read_sql_query("""
            SELECT 
                factor_type,
                CASE 
                    WHEN expression_name LIKE '%_l2%' THEN 'l2'
                    WHEN expression_name LIKE '%_l5%' THEN 'l5'
                    WHEN expression_name LIKE '%_l8%' THEN 'l8'
                    WHEN expression_name LIKE '%_l13%' THEN 'l13'
                    WHEN expression_name LIKE '%_l21%' THEN 'l21'
                    WHEN expression_name LIKE '%_l34%' THEN 'l34'
                    WHEN expression_name LIKE '%_l55%' THEN 'l55'
                    WHEN expression_name LIKE '%_l89%' THEN 'l89'
                    WHEN expression_name LIKE '%_l144%' THEN 'l144'
                    ELSE 'other'
                END as lookback,
                COUNT(*) as count,
                AVG(sharpe) as avg_sharpe,
                AVG(rank_ic) as avg_rank_ic
            FROM factor_evaluations 
            GROUP BY factor_type, lookback
            ORDER BY factor_type, lookback
        """, conn)
        print(df.to_string(index=False))
    
    conn.close()
    
    print("\n" + "="*80)
    print("查询完成！")
    print("="*80)

if __name__ == '__main__':
    main()
