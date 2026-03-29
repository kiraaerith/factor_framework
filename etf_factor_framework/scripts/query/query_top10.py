"""
查询 factor_eval.db 数据库前10条记录

使用方法:
    python query_top10.py
    
输出:
    - 终端显示前10条记录
    - 保存结果到 ../../../analysis/factor_eval_top10.txt
"""

import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime

# 数据库路径
DB_PATH = r"E:\code_project\factor_eval_result\factor_eval.db"

# 输出文件路径
# 脚本路径: etf_factor_framework\scripts\query\query_top10.py
# 目标路径: analysis\factor_eval_top10.txt (项目根目录下)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 从 scripts\query 回到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "analysis", "factor_eval_top10.txt")


def list_tables(conn):
    """列出数据库中所有表"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    return [t[0] for t in tables]


def get_table_schema(conn, table_name):
    """获取表结构"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return columns


def query_top10_records(conn, table_name):
    """查询前10条记录"""
    query = f"""
    SELECT * FROM {table_name} 
    LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    return df


def format_output(df, schema, table_name, total_count):
    """格式化输出"""
    lines = []
    lines.append("=" * 100)
    lines.append("Factor Evaluation Database - Top 10 Records")
    lines.append(f"Query Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database Path: {DB_PATH}")
    lines.append("=" * 100)
    lines.append("")
    
    # 显示表结构
    lines.append("-" * 100)
    lines.append(f"Table Schema ({table_name}):")
    lines.append("-" * 100)
    for col in schema:
        col_id, name, dtype, notnull, default, pk = col
        pk_mark = " (PK)" if pk else ""
        lines.append(f"  Column {col_id:2d}: {name:30s} | Type: {dtype:15s}{pk_mark}")
    lines.append("")
    
    # 显示前10条记录
    lines.append("-" * 100)
    lines.append("Top 10 Records:")
    lines.append("-" * 100)
    lines.append("")
    
    for idx, row in df.iterrows():
        lines.append(f"Record #{idx + 1} (ID: {row.get('id', 'N/A')})")
        lines.append("-" * 80)
        
        # 遍历所有列
        for col in df.columns:
            value = row[col]
            
            # 处理None值
            if value is None:
                value_str = "NULL"
            else:
                # 输出完整值，不截断
                value_str = str(value)
            
            # 对于长字符串，多行显示
            if isinstance(value, str) and len(value_str) > 100:
                lines.append(f"  {col:25s}: ")
                # 每行100字符换行显示
                for i in range(0, len(value_str), 100):
                    chunk = value_str[i:i+100]
                    lines.append(f"      {chunk}")
            else:
                lines.append(f"  {col:25s}: {value_str}")
        
        lines.append("")
    
    # 显示统计信息
    lines.append("-" * 100)
    lines.append("Summary Statistics:")
    lines.append("-" * 100)
    lines.append(f"  Total records displayed: {len(df)}")
    lines.append(f"  Total columns: {len(df.columns)}")
    lines.append(f"  Total records in database: {total_count}")
    lines.append("")
    
    # 显示列名列表
    lines.append("-" * 100)
    lines.append(f"All Columns ({len(df.columns)} total):")
    lines.append("-" * 100)
    for i, col in enumerate(df.columns, 1):
        lines.append(f"  {i:2d}. {col}")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("End of Report")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("Querying factor_eval.db - Top 10 Records")
    print("=" * 80)
    print()
    
    # 检查数据库是否存在
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)
    
    # 连接数据库
    try:
        conn = sqlite3.connect(DB_PATH)
        print(f"Connected to database: {DB_PATH}")
        print()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    try:
        # 列出所有表
        tables = list_tables(conn)
        print(f"Available tables: {tables}")
        print()
        
        if not tables:
            print("Error: No tables found in database")
            sys.exit(1)
        
        # 使用第一个表
        table_name = tables[0]
        print(f"Querying table: {table_name}")
        
        # 获取表结构
        schema = get_table_schema(conn, table_name)
        print(f"Table '{table_name}' has {len(schema)} columns")
        
        # 查询前10条记录
        df = query_top10_records(conn, table_name)
        print(f"Retrieved {len(df)} records")
        
        # 获取总记录数
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_count = cursor.fetchone()[0]
        print(f"Total records in table: {total_count}")
        print()
        
        # 格式化输出
        output = format_output(df, schema, table_name, total_count)
        
        # 在终端显示
        print(output)
        
        # 保存到文件
        # 确保输出目录存在
        output_dir = os.path.dirname(OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print()
        print("=" * 80)
        print(f"Results saved to: {OUTPUT_PATH}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error querying database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
