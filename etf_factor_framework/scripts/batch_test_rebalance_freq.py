#!/usr/bin/env python3
"""
批量测试不同调仓频率的脚本

使用同一个因子配置，测试多个 rebalance_freq 值。
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
from itertools import product

from config import load_config, Config, ConfigValidator
from core import OHLCVData
from factors import create_factor
from mappers import RankBasedMapper
from evaluation import FactorEvaluator
from storage import ResultStorage, StorageConfig


def run_single_evaluation(config: Config, rebalance_freq: int, storage: ResultStorage = None):
    """
    运行单次评估
    
    Args:
        config: 基础配置
        rebalance_freq: 调仓频率
        storage: 存储对象（可选）
        
    Returns:
        dict: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"测试调仓频率: {rebalance_freq}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print("1. 加载数据...")
    df = pd.read_csv(config.data.csv_path)
    df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])
    df.columns = [c.lower() for c in df.columns]
    
    ohlcv = OHLCVData.from_dataframe(
        df,
        symbol_col=config.data.symbol_col,
        date_col=config.data.date_col,
        ohlcv_cols=config.data.ohlcv_cols
    )
    print(f"   数据形状: {ohlcv.to_dataframe().shape}")
    
    # 2. 计算因子
    factor_config = config.factors[0]
    print(f"2. 计算因子: {factor_config.name} ({factor_config.type})...")
    
    factor = create_factor(factor_config.type, factor_config.params)
    factor_data = factor.compute(ohlcv)
    print(f"   因子形状: {factor_data.values.shape}")
    
    # 3. 创建映射器
    print(f"3. 创建仓位映射器 ({config.mapper.type})...")
    if config.mapper.type == 'rank_based':
        mapper = RankBasedMapper(**config.mapper.params)
    else:
        raise ValueError(f"不支持的映射器类型: {config.mapper.type}")
    
    position_data = mapper.map_to_position(factor_data)
    print(f"   仓位形状: {position_data.weights.shape}")
    
    # 4. 运行评估（使用指定的 rebalance_freq）
    print(f"4. 运行评估 (rebalance_freq={rebalance_freq})...")
    evaluator = FactorEvaluator(
        forward_period=config.evaluation.forward_period,
        periods_per_year=config.evaluation.periods_per_year,
        risk_free_rate=config.evaluation.risk_free_rate,
        commission_rate=config.evaluation.commission_rate,
        slippage_rate=config.evaluation.slippage_rate,
        delay=config.evaluation.delay,
        rebalance_freq=rebalance_freq,  # 使用传入的值
    )
    
    results = evaluator.evaluate(factor_data, ohlcv, position_data)
    
    # 打印关键指标
    ic_metrics = results.get('ic_metrics', {})
    risk_adj_metrics = results.get('risk_adjusted_metrics', {})
    print(f"   Rank IC: {ic_metrics.get('rank_ic_mean', 0):.4f}")
    print(f"   Sharpe:  {risk_adj_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"   Calmar:  {risk_adj_metrics.get('calmar_ratio', 0):.4f}")
    
    # 5. 保存结果
    if storage and storage._is_database_mode:
        print("5. 保存结果到数据库...")
        
        # 构建 dataset_name，包含 rebalance_freq 信息
        csv_path = config.data.csv_path
        dataset_name = f"{Path(csv_path).stem}_rebal{rebalance_freq}"
        
        record_id = storage.save_evaluation_result(
            result=results,
            factor_name=factor_config.name,
            factor_type=factor_config.type,
            params=factor_config.params,
            dataset_name=dataset_name,
            dataset_params={
                'config_name': config.name,
                'config_version': config.version,
                'csv_path': csv_path,
                'rebalance_freq': rebalance_freq,
                'evaluated_at': datetime.now().isoformat(),
            }
        )
        print(f"   记录ID: {record_id}")
    
    return results


def batch_test_rebalance_freq(
    config_path: str,
    rebalance_freqs: list = None,
    db_path: str = None
):
    """
    批量测试不同调仓频率
    
    Args:
        config_path: 基础配置文件路径
        rebalance_freqs: 调仓频率列表，默认为 [5, 10, 20]
        db_path: 数据库路径（可选）
    """
    if rebalance_freqs is None:
        rebalance_freqs = [5, 10, 20]
    
    print(f"\n{'#'*70}")
    print(f"# 批量测试调仓频率")
    print(f"# 配置文件: {config_path}")
    print(f"# 测试频率: {rebalance_freqs}")
    print(f"# 数据库: {db_path or '文件模式'}")
    print(f"{'#'*70}\n")
    
    # 加载基础配置
    config = load_config(config_path)
    
    # 验证配置
    validator = ConfigValidator()
    if not validator.validate(config):
        print("配置验证失败:")
        for error in validator.errors:
            print(f"  - {error}")
        return
    
    # 创建存储对象
    storage = None
    if db_path:
        storage_config = StorageConfig(
            storage_mode='database',
            db_path=db_path,
            base_path=r"E:\code_project\factor_eval_result\etf",
        )
        storage = ResultStorage(storage_config)
    
    # 批量运行
    all_results = []
    for freq in rebalance_freqs:
        try:
            results = run_single_evaluation(config, freq, storage)
            all_results.append({
                'rebalance_freq': freq,
                'results': results,
            })
        except Exception as e:
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总
    print(f"\n{'='*70}")
    print("汇总结果")
    print(f"{'='*70}")
    print(f"{'调仓频率':<10} {'Rank IC':<12} {'Sharpe':<12} {'Calmar':<12}")
    print("-" * 70)
    for item in all_results:
        freq = item['rebalance_freq']
        results = item['results']
        ic = results.get('ic_metrics', {}).get('rank_ic_mean', 0)
        sharpe = results.get('risk_adjusted_metrics', {}).get('sharpe_ratio', 0)
        calmar = results.get('risk_adjusted_metrics', {}).get('calmar_ratio', 0)
        print(f"{freq:<10} {ic:<12.4f} {sharpe:<12.4f} {calmar:<12.4f}")
    
    print(f"\n完成！共测试 {len(all_results)} 个调仓频率。")
    return all_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量测试不同调仓频率')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--freqs', '-f', nargs='+', type=int, 
                        default=[5, 10, 20],
                        help='调仓频率列表，默认为 5 10 20')
    parser.add_argument('--db', '-d', 
                        default=str(Path(__file__).resolve().parent.parent.parent / "factor_eval_result" / "factor_eval.db"),
                        help='数据库路径')
    parser.add_argument('--no-db', action='store_true',
                        help='不使用数据库，仅文件存储')
    
    args = parser.parse_args()
    
    db_path = None if args.no_db else args.db
    
    batch_test_rebalance_freq(
        config_path=args.config,
        rebalance_freqs=args.freqs,
        db_path=db_path
    )


if __name__ == '__main__':
    main()
