"""
基本面因子参数网格结果查询脚本

按因子名称、中性化方式、持仓股数、持仓周期筛选评估结果，
支持按 IC IR / Rank IC IR / 夏普等指标排序展示。

使用示例：
  python query_fundamental_grid.py
  python query_fundamental_grid.py --factor ROE --sort rank_icir
  python query_fundamental_grid.py --factor ROE --neutral industry --top_k 50
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# 默认数据库路径（自动检测：脚本 scripts/query/ → scripts/ → etf_factor_framework/ → 项目根目录）
_SCRIPT_DIR = Path(__file__).resolve().parent  # .../etf_factor_framework/scripts/query/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent  # .../etf_cross_ml-master/
_AUTO_DB_PATH = _PROJECT_ROOT / "factor_eval_result" / "factor_eval.db"
DEFAULT_DB_PATH = str(_AUTO_DB_PATH) if _AUTO_DB_PATH.exists() else r"D:\code_project_v2\factor_eval_result\factor_eval.db"


def query_results(
    db_path: str = DEFAULT_DB_PATH,
    factor_name: str = None,
    neutral_method: str = None,
    top_k: int = None,
    rebalance_freq: int = None,
    sort_by: str = "rank_icir",
    limit: int = 100,
) -> pd.DataFrame:
    """
    查询基本面因子网格评估结果。

    Args:
        db_path: SQLite 数据库路径
        factor_name: 因子名称（模糊匹配，如 'ROE'）
        neutral_method: 中性化方式 ('raw' / 'industry' / 'size')
        top_k: 持仓股数筛选
        rebalance_freq: 调仓频率筛选
        sort_by: 排序字段（rank_icir / icir / sharpe / calmar）
        limit: 返回行数上限

    Returns:
        DataFrame: 查询结果
    """
    valid_sort = {"rank_icir", "icir", "sharpe", "calmar", "rank_ic", "ic", "max_drawdown",
                   "excess_ret_csi300", "excess_ret_csi500", "excess_ret_csi2000",
                   "ir_csi300", "ir_csi500", "ir_csi2000"}
    if sort_by not in valid_sort:
        raise ValueError(f"sort_by must be one of {valid_sort}")

    conditions = []
    params = []

    if factor_name:
        conditions.append("expression_name LIKE ?")
        params.append(f"%{factor_name}%")

    if neutral_method:
        conditions.append("neutralization_method = ?")
        params.append(neutral_method)

    if top_k is not None:
        conditions.append("top_k = ?")
        params.append(top_k)

    if rebalance_freq is not None:
        conditions.append("rebalance_freq = ?")
        params.append(rebalance_freq)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    asc_desc = "ASC" if sort_by == "max_drawdown" else "DESC"

    sql = f"""
    SELECT
        expression_name,
        neutralization_method,
        top_k,
        rebalance_freq,
        forward_period,
        ic,
        icir,
        rank_ic,
        rank_icir,
        sharpe,
        max_drawdown,
        calmar,
        excess_ret_csi300,
        excess_ret_csi500,
        excess_ret_csi2000,
        ir_csi300,
        ir_csi500,
        ir_csi2000,
        created_at
    FROM factor_evaluation_results
    {where_clause}
    ORDER BY {sort_by} {asc_desc}
    LIMIT {limit}
    """

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def print_table(df: pd.DataFrame, sort_by: str):
    """格式化打印查询结果表格"""
    if df.empty:
        print("  (无结果)")
        return

    # 格式化数值列
    fmt = {
        "ic":         lambda x: f"{x:.4f}" if pd.notna(x) else "  N/A",
        "icir":       lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "rank_ic":    lambda x: f"{x:.4f}" if pd.notna(x) else "  N/A",
        "rank_icir":  lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "sharpe":     lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "max_drawdown": lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "calmar":     lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "excess_ret_csi300":  lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "excess_ret_csi500":  lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "excess_ret_csi2000": lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "ir_csi300":  lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "ir_csi500":  lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
        "ir_csi2000": lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A",
    }

    display = df.copy()
    for col, fn in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(fn)

    # 截断 expression_name
    display["expression_name"] = display["expression_name"].apply(
        lambda x: x[:30] if isinstance(x, str) else x
    )
    display = display.drop(columns=["created_at"], errors="ignore")

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(display.to_string(index=False))
    print(f"\n共 {len(df)} 条记录（排序: {sort_by}）")


def summary_by_neutral(df: pd.DataFrame):
    """按中性化方式汇总均值"""
    if df.empty or "neutralization_method" not in df.columns:
        return
    grp = df.groupby("neutralization_method")[["icir", "rank_icir", "sharpe", "calmar"]].mean()
    print("\n--- 按中性化方式均值汇总 ---")
    print(grp.round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description="查询基本面因子网格评估结果")
    parser.add_argument("--db",      default=DEFAULT_DB_PATH, help="数据库路径")
    parser.add_argument("--factor",  default=None,  help="因子名称（模糊匹配）")
    parser.add_argument("--neutral", default=None,  help="中性化方式: raw/industry/size")
    parser.add_argument("--top_k",   type=int, default=None, help="持仓股数")
    parser.add_argument("--freq",    type=int, default=None, help="调仓频率")
    parser.add_argument("--sort",    default="rank_icir",    help="排序字段（默认 rank_icir）")
    parser.add_argument("--limit",   type=int, default=100,  help="最大返回行数")
    parser.add_argument("--summary", action="store_true",    help="同时输出分组均值汇总")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"[ERROR] 数据库不存在: {args.db}")
        sys.exit(1)

    print(f"查询数据库: {args.db}")
    filters = []
    if args.factor:  filters.append(f"factor={args.factor}")
    if args.neutral: filters.append(f"neutral={args.neutral}")
    if args.top_k:   filters.append(f"top_k={args.top_k}")
    if args.freq:    filters.append(f"freq={args.freq}")
    print(f"筛选条件: {', '.join(filters) if filters else '(全部)'}")
    print()

    df = query_results(
        db_path=args.db,
        factor_name=args.factor,
        neutral_method=args.neutral,
        top_k=args.top_k,
        rebalance_freq=args.freq,
        sort_by=args.sort,
        limit=args.limit,
    )

    print_table(df, args.sort)

    if args.summary:
        summary_by_neutral(df)


if __name__ == "__main__":
    main()
