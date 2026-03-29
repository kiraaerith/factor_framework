"""
RSI 因子评估并保存到数据库

演示如何使用数据库存储模式评估 RSI 因子。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from core.ohlcv_data import OHLCVData
from core.factor_data import FactorData
from core.position_data import PositionData
from factors.technical_factors import RSI
from mappers.position_mappers import RankBasedMapper
from evaluation import FactorEvaluator
from storage import ResultStorage, StorageConfig, DatabaseStorage


def load_etf_data(csv_path: str = None) -> OHLCVData:
    """
    加载 ETF 数据
    
    Args:
        csv_path: CSV 文件路径，默认使用项目根目录的 etf_rotation_daily.csv
        
    Returns:
        OHLCVData: OHLCV 数据对象
    """
    if csv_path is None:
        # 默认路径：项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(project_root, "etf_rotation_daily.csv")
    
    print(f"正在加载数据: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['eob'] = pd.to_datetime(df['eob'])
    
    print(f"  原始数据行数: {len(df)}")
    print(f"  时间范围: {df['eob'].min()} ~ {df['eob'].max()}")
    
    # 构建 OHLCVData
    ohlcv = OHLCVData.from_dataframe(
        df,
        symbol_col='symbol',
        date_col='eob',
        ohlcv_cols={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
    )
    
    print(f"  ETF数量: {ohlcv.n_assets}")
    print(f"  时间长度: {ohlcv.n_periods}")
    print(f"  ETF列表: {', '.join(ohlcv.symbols[:5])}{'...' if ohlcv.n_assets > 5 else ''}")
    
    return ohlcv


def evaluate_rsi(
    ohlcv: OHLCVData,
    rsi_period: int = 14,
    top_k: int = 5,
    direction: int = -1,
    forward_period: int = 5,
    dataset_name: str = None
) -> dict:
    """
    评估 RSI 因子
    
    Args:
        ohlcv: OHLCV 数据
        rsi_period: RSI 计算周期
        top_k: 选股数量
        direction: 方向（1=做多，-1=做空）
        forward_period: 前瞻期数
        dataset_name: 数据集名称
        
    Returns:
        dict: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"RSI 因子评估")
    print(f"{'='*60}")
    print(f"  RSI周期: {rsi_period}")
    print(f"  TopK: {top_k}")
    print(f"  方向: {'做多' if direction == 1 else '做空'}")
    print(f"  前瞻期: {forward_period}")
    
    # 1. 计算 RSI 因子
    print(f"\n[1/4] 计算 RSI 因子...")
    rsi_calc = RSI(period=rsi_period, field='close')
    factor_data = rsi_calc.calculate(ohlcv)
    print(f"  因子名称: {factor_data.name}")
    print(f"  因子形状: {factor_data.shape}")
    
    # 2. 仓位映射
    print(f"\n[2/4] 仓位映射 (RankBased)...")
    mapper = RankBasedMapper(
        top_k=top_k,
        direction=direction,
        weight_method='equal'
    )
    position_data = mapper.map_to_position(factor_data)
    print(f"  仓位形状: {position_data.shape}")
    print(f"  日均持仓数: {position_data.weights.sum(axis=0).mean():.2f}")
    
    # 3. 创建评估器
    print(f"\n[3/4] 创建评估器...")
    evaluator = FactorEvaluator(
        factor_data=factor_data,
        ohlcv_data=ohlcv,
        position_data=position_data,
        forward_period=forward_period,
        periods_per_year=252,
        risk_free_rate=0.03,
        commission_rate=0.0002,
        slippage_rate=0.0,
        delay=1,
        rebalance_freq=1,
    )
    
    # 4. 运行评估
    print(f"\n[4/4] 运行评估...")
    results = evaluator.run_full_evaluation()
    
    # 打印关键指标
    print(f"\n{'-'*60}")
    print("评估结果")
    print(f"{'-'*60}")
    
    ic_metrics = results.get('ic_metrics', {})
    print(f"IC指标:")
    print(f"  IC Mean:       {ic_metrics.get('ic_mean', 0):.4f}")
    print(f"  IC IR:         {ic_metrics.get('ic_ir', 0):.4f}")
    print(f"  Rank IC Mean:  {ic_metrics.get('rank_ic_mean', 0):.4f}")
    print(f"  Rank IC IR:    {ic_metrics.get('rank_ic_ir', 0):.4f}")
    
    returns_metrics = results.get('returns_metrics', {})
    print(f"收益指标:")
    print(f"  总收益:        {returns_metrics.get('total_return', 0):.2%}")
    print(f"  年化收益:      {returns_metrics.get('annualized_return', 0):.2%}")
    
    risk_metrics = results.get('risk_metrics', {})
    print(f"风险指标:")
    print(f"  最大回撤:      {risk_metrics.get('max_drawdown', 0):.2%}")
    print(f"  年化波动率:    {risk_metrics.get('annualized_volatility', 0):.2%}")
    
    risk_adj_metrics = results.get('risk_adjusted_metrics', {})
    print(f"风险调整指标:")
    print(f"  夏普比率:      {risk_adj_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Calmar比率:    {risk_adj_metrics.get('calmar_ratio', 0):.4f}")
    print(f"  Sortino比率:   {risk_adj_metrics.get('sortino_ratio', 0):.4f}")
    
    turnover_metrics = results.get('turnover_metrics', {})
    print(f"换手率指标:")
    print(f"  日均换手率:    {turnover_metrics.get('avg_daily_turnover', 0):.2%}")
    print(f"  年化换手率:    {turnover_metrics.get('annualized_turnover', 0):.2%}")
    
    return {
        'results': results,
        'evaluator': evaluator,
        'factor_data': factor_data,
        'position_data': position_data,
        'dataset_name': dataset_name or 'ETF_Rotation_Daily',
    }


def save_to_database(
    eval_info: dict,
    db_path: str = r"E:\code_project\factor_eval_result\factor_eval.db"
):
    """
    保存评估结果到数据库
    
    Args:
        eval_info: 评估信息字典
        db_path: 数据库路径
    """
    print(f"\n{'='*60}")
    print("保存到数据库")
    print(f"{'='*60}")
    print(f"  数据库路径: {db_path}")
    
    # 创建数据库配置
    storage_config = StorageConfig(
        storage_mode='database',
        db_path=db_path
    )
    
    # 创建存储器
    storage = ResultStorage(storage_config)
    
    # 获取评估结果
    results = eval_info['results']
    factor_data = eval_info['factor_data']
    
    # 提取因子参数
    factor_params = results.get('factor_params', {})
    if not factor_params and hasattr(factor_data, 'params'):
        factor_params = factor_data.params
    
    # 保存到数据库
    record_id = storage.save_evaluation_result(
        result=results,
        factor_name=results.get('factor_name', 'RSI'),
        params=factor_params,
        dataset_name=eval_info['dataset_name'],
        dataset_params={
            'source': 'etf_rotation_daily.csv',
            'evaluated_at': datetime.now().isoformat(),
        }
    )
    
    print(f"\n  保存成功!")
    print(f"  记录ID: {record_id}")
    
    # 验证保存结果
    db = DatabaseStorage(db_path)
    stats = db.get_statistics()
    print(f"\n  数据库统计:")
    print(f"    总记录数: {stats['total_records']}")
    print(f"    不同表达式数: {stats['distinct_expressions']}")
    print(f"    不同数据集数: {stats['distinct_datasets']}")
    
    # 查询刚保存的记录
    df = db.query_by_expression_name(results.get('factor_name', 'RSI'), limit=1)
    if not df.empty:
        print(f"\n  最新记录:")
        print(f"    表达式: {df.iloc[0]['expression_name']}")
        print(f"    数据集: {df.iloc[0]['dataset_name']}")
        print(f"    夏普: {df.iloc[0]['sharpe']:.4f}")
        print(f"    Rank IC: {df.iloc[0]['rank_ic']:.4f}")
        print(f"    创建时间: {df.iloc[0]['created_at']}")
    
    return record_id


def query_and_display(db_path: str = r"E:\code_project\factor_eval_result\factor_eval.db"):
    """
    查询并展示数据库中的 RSI 评估结果
    
    Args:
        db_path: 数据库路径
    """
    print(f"\n{'='*60}")
    print("数据库查询演示")
    print(f"{'='*60}")
    
    db = DatabaseStorage(db_path)
    
    # 1. 按表达式名称查询
    print(f"\n[1] 按表达式名称查询 'RSI':")
    df = db.query_by_expression_name('RSI')
    print(f"    找到 {len(df)} 条记录")
    if not df.empty:
        print(f"    最新记录 Rank IC: {df.iloc[0]['rank_ic']:.4f}, Sharpe: {df.iloc[0]['sharpe']:.4f}")
    
    # 2. 按指标范围查询
    print(f"\n[2] 按夏普比率查询 (> 0):")
    df = db.query_by_metric_range('sharpe', min_value=0)
    print(f"    找到 {len(df)} 条记录")
    
    # 3. 按表达式名和参数查询
    print(f"\n[3] 按表达式名和参数查询 (period=14):")
    df = db.query_by_expression_and_params('RSI', {'period': 14})
    print(f"    找到 {len(df)} 条记录")
    
    # 4. 获取所有不同的表达式
    print(f"\n[4] 所有表达式名称:")
    expressions = db.get_distinct_expressions()
    for expr in expressions:
        print(f"    - {expr}")
    
    # 5. 打印完整统计
    print(f"\n[5] 数据库统计:")
    stats = db.get_statistics()
    print(f"    总记录数: {stats['total_records']}")
    print(f"    不同表达式: {stats['distinct_expressions']}")
    print(f"    不同数据集: {stats['distinct_datasets']}")
    
    if stats['metrics_statistics']:
        print(f"\n    指标统计:")
        for metric, stat in stats['metrics_statistics'].items():
            if stat['count'] > 0:
                print(f"      {metric}: 均值={stat['avg']:.4f}, 最大={stat['max']:.4f}, 最小={stat['min']:.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RSI 因子评估并保存到数据库')
    parser.add_argument('--period', type=int, default=14, help='RSI 周期 (默认: 14)')
    parser.add_argument('--top-k', type=int, default=5, help='选股数量 (默认: 5)')
    parser.add_argument('--direction', type=int, default=-1, help='方向: 1=做多, -1=做空 (默认: -1)')
    parser.add_argument('--forward', type=int, default=5, help='前瞻期数 (默认: 5)')
    parser.add_argument('--csv', type=str, default=None, help='CSV 数据文件路径')
    parser.add_argument('--db-path', type=str, 
                        default=r"E:\code_project\factor_eval_result\factor_eval.db",
                        help='数据库路径')
    parser.add_argument('--query-only', action='store_true', help='仅查询数据库，不运行评估')
    
    args = parser.parse_args()
    
    if args.query_only:
        # 仅查询模式
        query_and_display(args.db_path)
        return
    
    # 完整评估流程
    print("="*70)
    print("RSI 因子评估 - 数据库存储模式")
    print("="*70)
    
    # 1. 加载数据
    ohlcv = load_etf_data(args.csv)
    
    # 2. 评估 RSI 因子
    eval_info = evaluate_rsi(
        ohlcv=ohlcv,
        rsi_period=args.period,
        top_k=args.top_k,
        direction=args.direction,
        forward_period=args.forward,
        dataset_name='ETF_Rotation_Daily'
    )
    
    # 3. 保存到数据库
    record_id = save_to_database(eval_info, args.db_path)
    
    # 4. 查询演示
    query_and_display(args.db_path)
    
    print(f"\n{'='*70}")
    print("评估完成！")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
