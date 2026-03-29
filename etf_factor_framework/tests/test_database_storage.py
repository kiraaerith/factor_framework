"""
数据库存储功能测试脚本

测试 SQLite 数据库存储模式的各项功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from core.factor_data import FactorData
from core.position_data import PositionData
from core.ohlcv_data import OHLCVData
from factors.technical_factors import CloseOverMA, RSI
from mappers.position_mappers import RankBasedMapper
from evaluation import FactorEvaluator

from config.base_config import StorageConfig
from storage import ResultStorage, DatabaseStorage


def generate_sample_data(n_days: int = 252):
    """生成示例数据"""
    np.random.seed(42)
    
    symbols = ['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # 生成价格数据
    close_prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(5, n_days) * 0.02, axis=1) * 10,
        index=symbols,
        columns=dates
    )
    
    open_prices = close_prices * (1 + np.random.randn(5, n_days) * 0.005)
    high_prices = pd.concat([open_prices, close_prices]).groupby(level=0).max()
    high_prices = high_prices * (1 + np.abs(np.random.randn(5, n_days) * 0.01))
    low_prices = pd.concat([open_prices, close_prices]).groupby(level=0).min()
    low_prices = low_prices * (1 - np.abs(np.random.randn(5, n_days) * 0.01))
    volume = pd.DataFrame(
        np.abs(np.random.randn(5, n_days)) * 1000000,
        index=symbols,
        columns=dates
    )
    
    return OHLCVData(
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume
    )


def test_database_storage_mode():
    """测试数据库存储模式"""
    print("=" * 60)
    print("测试1: 数据库存储模式")
    print("=" * 60)
    
    # 创建数据库配置
    db_path = r"E:\code_project\factor_eval_result\test_factor_eval.db"
    config = StorageConfig(
        storage_mode='database',
        db_path=db_path
    )
    
    print(f"\n存储模式: {config.storage_mode}")
    print(f"数据库路径: {config.db_path}")
    
    # 创建存储器
    storage = ResultStorage(config)
    
    # 生成数据
    ohlcv = generate_sample_data(252)
    
    # 计算因子和仓位
    factor_calc = CloseOverMA(period=20)
    factor_data = factor_calc.calculate(ohlcv)
    
    mapper = RankBasedMapper(top_k=3, weight_method='equal', direction=-1)
    position_data = mapper.map_to_position(factor_data)
    
    # 创建评估器
    evaluator = FactorEvaluator(
        factor_data=factor_data,
        ohlcv_data=ohlcv,
        position_data=position_data,
        forward_period=5,
    )
    
    # 运行评估
    result = evaluator.run_full_evaluation()
    
    # 保存到数据库
    record_id = storage.save_evaluation_result(
        result=result,
        factor_name='CloseOverMA',
        params={'period': 20},
        dataset_name='ETF_Daily_2023',
        dataset_params={'start_date': '2023-01-01', 'end_date': '2023-12-31'}
    )
    
    print(f"\n评估结果已保存到数据库，记录ID: {record_id}")
    
    # 查询验证
    db = DatabaseStorage(db_path)
    
    print("\n按表达式名称查询:")
    df = db.query_by_expression_name('CloseOverMA')
    print(f"  找到 {len(df)} 条记录")
    if not df.empty:
        print(f"  夏普比率: {df.iloc[0]['sharpe']:.4f}")
        print(f"  Rank IC: {df.iloc[0]['rank_ic']:.4f}")
    
    print("\n按数据集名称查询:")
    df = db.query_by_dataset_name('ETF_Daily_2023')
    print(f"  找到 {len(df)} 条记录")
    
    print("\n按指标范围查询 (sharpe > 0):")
    df = db.query_by_metric_range('sharpe', min_value=0)
    print(f"  找到 {len(df)} 条记录")
    
    # 统计信息
    print("\n数据库统计信息:")
    stats = db.get_statistics()
    print(f"  总记录数: {stats['total_records']}")
    print(f"  不同表达式数: {stats['distinct_expressions']}")
    print(f"  不同数据集数: {stats['distinct_datasets']}")
    
    print("\n测试1通过!")
    return db_path


def test_file_storage_mode():
    """测试文件存储模式"""
    print("\n" + "=" * 60)
    print("测试2: 文件存储模式")
    print("=" * 60)
    
    # 创建文件配置
    config = StorageConfig(
        storage_mode='file',
        base_path=r"E:\code_project\factor_eval_result\test_etf_file"
    )
    
    print(f"\n存储模式: {config.storage_mode}")
    print(f"基础路径: {config.base_path}")
    
    # 创建存储器
    storage = ResultStorage(config)
    
    # 生成数据
    ohlcv = generate_sample_data(252)
    
    # 计算因子和仓位
    factor_calc = RSI(period=14)
    factor_data = factor_calc.calculate(ohlcv)
    
    mapper = RankBasedMapper(top_k=3, weight_method='equal', direction=1)
    position_data = mapper.map_to_position(factor_data)
    
    # 创建评估器
    evaluator = FactorEvaluator(
        factor_data=factor_data,
        ohlcv_data=ohlcv,
        position_data=position_data,
        forward_period=5,
    )
    
    # 运行评估并生成图表
    result = evaluator.run_full_evaluation()
    figures = evaluator.plot()
    report = evaluator.generate_report()
    
    # 保存到文件
    saved_files = storage.save_evaluation_result(
        result=result,
        figures=figures,
        report=report,
        factor_name='RSI',
        params={'period': 14}
    )
    
    print(f"\n评估结果已保存到文件:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")
    
    print("\n测试2通过!")
    return config.base_path


def test_query_functions(db_path: str):
    """测试查询功能"""
    print("\n" + "=" * 60)
    print("测试3: 查询功能")
    print("=" * 60)
    
    db = DatabaseStorage(db_path)
    
    # 添加更多测试数据
    ohlcv = generate_sample_data(252)
    
    for period in [5, 10, 20]:
        factor_calc = CloseOverMA(period=period)
        factor_data = factor_calc.calculate(ohlcv)
        
        mapper = RankBasedMapper(top_k=3, weight_method='equal', direction=-1)
        position_data = mapper.map_to_position(factor_data)
        
        evaluator = FactorEvaluator(
            factor_data=factor_data,
            ohlcv_data=ohlcv,
            position_data=position_data,
            forward_period=5,
        )
        
        result = evaluator.run_full_evaluation()
        
        db.save_evaluation_result(
            expression_name='CloseOverMA',
            dataset_name='ETF_Daily_2023',
            result=result,
            expression_params={'period': period},
            dataset_params={'start_date': '2023-01-01'}
        )
    
    print("\n添加了3条不同参数的记录")
    
    # 按表达式名称和参数查询
    print("\n按表达式名称和参数查询 (period=10):")
    df = db.query_by_expression_and_params(
        expression_name='CloseOverMA',
        expression_params={'period': 10}
    )
    print(f"  找到 {len(df)} 条记录")
    
    # 按指标范围查询
    print("\n按 Calmar 比率范围查询 (> 0.5):")
    df = db.query_by_metric_range('calmar', min_value=0.5)
    print(f"  找到 {len(df)} 条记录")
    if not df.empty:
        print(f"  Calmar 范围: {df['calmar'].min():.4f} ~ {df['calmar'].max():.4f}")
    
    # 获取不同的表达式和数据集
    print("\n所有表达式名称:")
    expressions = db.get_distinct_expressions()
    for expr in expressions:
        print(f"  - {expr}")
    
    print("\n所有数据集名称:")
    datasets = db.get_distinct_datasets()
    for ds in datasets:
        print(f"  - {ds}")
    
    print("\n测试3通过!")


def test_config_switching():
    """测试配置切换"""
    print("\n" + "=" * 60)
    print("测试4: 配置切换验证")
    print("=" * 60)
    
    # 文件模式配置
    file_config = StorageConfig(storage_mode='file')
    print(f"\n文件模式: is_file_mode={file_config.is_file_mode()}, is_database_mode={file_config.is_database_mode()}")
    assert file_config.is_file_mode()
    assert not file_config.is_database_mode()
    
    # 数据库模式配置
    db_config = StorageConfig(storage_mode='database')
    print(f"数据库模式: is_file_mode={db_config.is_file_mode()}, is_database_mode={db_config.is_database_mode()}")
    assert not db_config.is_file_mode()
    assert db_config.is_database_mode()
    
    # 无效配置测试
    try:
        invalid_config = StorageConfig(storage_mode='invalid')
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"\n无效模式正确抛出异常: {e}")
    
    print("\n测试4通过!")


def cleanup_test_files(db_path: str, file_base_path: str):
    """清理测试文件"""
    print("\n" + "=" * 60)
    print("清理测试文件")
    print("=" * 60)
    
    import shutil
    
    # 删除测试数据库
    if Path(db_path).exists():
        Path(db_path).unlink()
        print(f"已删除测试数据库: {db_path}")
    
    # 删除测试文件目录
    if Path(file_base_path).exists():
        shutil.rmtree(file_base_path)
        print(f"已删除测试目录: {file_base_path}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("因子评估数据库存储功能测试")
    print("=" * 60)
    
    # 运行所有测试
    db_path = test_database_storage_mode()
    file_path = test_file_storage_mode()
    test_query_functions(db_path)
    test_config_switching()
    
    # 清理
    cleanup_test_files(db_path, file_path)
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
