"""
使用配置文件运行因子评估示例

演示如何使用配置系统来运行端到端的因子评估流程。
支持文件存储和数据库存储两种模式，通过配置文件中的 storage.storage_mode 控制。
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from config import load_config, Config, ConfigValidator, StorageConfig as BaseStorageConfig
from core import OHLCVData
from factors import create_factor
from mappers import RankBasedMapper, DirMapper, QuantileMapper, ZScoreMapper
from evaluation import FactorEvaluator
from storage import ResultStorage, StorageConfig, DatabaseStorage


def get_dataset_name_from_config(config) -> str:
    """
    从配置中提取数据集名称

    - CSV模式：使用文件名（不含扩展名）
    - DuckDB模式：使用 "astock_<start>_<end>" 或配置名称

    Args:
        config: 配置对象

    Returns:
        str: 数据集名称
    """
    data_source = getattr(config.data, 'data_source', 'csv')

    if data_source == 'csv':
        csv_path = getattr(config.data, 'csv_path', None)
        if csv_path:
            dataset_name = Path(csv_path).stem
            if dataset_name:
                return dataset_name
    elif data_source == 'duckdb':
        start = getattr(config.data, 'start_date', '') or ''
        end = getattr(config.data, 'end_date', '') or ''
        if start and end:
            return f"astock_{start[:7]}_{end[:7]}"
        return 'astock'

    if hasattr(config, 'name') and config.name:
        return config.name

    return 'unknown_dataset'


def load_ohlcv_from_config(data_config, config_dir: str = None) -> OHLCVData:
    """从CSV文件加载OHLCV数据（ETF模式，向后兼容）"""
    csv_path = data_config.csv_path

    # 如果是相对路径，尝试基于配置文件目录解析
    if not os.path.isabs(csv_path) and config_dir:
        csv_path = os.path.normpath(os.path.join(config_dir, csv_path))

    print(f"正在加载数据: {csv_path}")

    df = pd.read_csv(csv_path)
    df[data_config.date_col] = pd.to_datetime(df[data_config.date_col])
    df.columns = [c.lower() for c in df.columns]

    ohlcv = OHLCVData.from_dataframe(
        df,
        symbol_col=data_config.symbol_col.lower(),
        date_col=data_config.date_col.lower(),
        ohlcv_cols=data_config.ohlcv_cols,
    )

    print(f"  - ETF数量: {ohlcv.n_assets}")
    print(f"  - 时间长度: {ohlcv.n_periods}")

    return ohlcv


def load_ohlcv_from_duckdb(data_config):
    """
    从DuckDB加载A股OHLCV数据及交易上下文

    Returns:
        tuple: (OHLCVData, TradeContext 或 None)
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.stock_data_loader import StockDataLoader

    loader = StockDataLoader()

    start_date = data_config.start_date
    end_date = data_config.end_date
    if not start_date or not end_date:
        raise ValueError(
            "DuckDB数据源需要在配置中指定 start_date 和 end_date"
        )

    use_adjusted = getattr(data_config, 'use_adjusted', True)
    lookback_extra_days = getattr(data_config, 'lookback_extra_days', 120)
    board_filter = getattr(data_config, 'board_filter', None)
    new_stock_filter_days = getattr(data_config, 'new_stock_filter_days', 365)

    print(f"正在加载数据 (数据源: tushare)")
    print(f"  日期范围: {start_date} ~ {end_date}")

    # 加载后复权OHLCV
    ohlcv = loader.load_ohlcv(
        start_date=start_date,
        end_date=end_date,
        use_adjusted=use_adjusted,
        lookback_extra_days=lookback_extra_days,
    )

    # 加载不复权开盘价（用于涨跌停判断）
    symbols = ohlcv.symbols
    raw_open = loader.load_raw_open(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
    )

    # 构建交易上下文
    print("  正在构建交易上下文（停牌/涨跌停掩码）...")
    trade_context = loader.load_trade_context(
        start_date=start_date,
        end_date=end_date,
        raw_open=raw_open,
        symbols=symbols,
        new_stock_filter_days=new_stock_filter_days,
        suspended_value_mode='freeze',
    )

    loader.close()
    return ohlcv, trade_context


def load_data_from_config(data_config, config_dir: str = None):
    """
    根据配置的 data_source 字段选择数据加载方式

    Returns:
        tuple: (OHLCVData, TradeContext 或 None)
    """
    data_source = getattr(data_config, 'data_source', 'csv')

    if data_source == 'duckdb':
        return load_ohlcv_from_duckdb(data_config)
    else:
        # CSV 模式（原有 ETF 流程）
        return load_ohlcv_from_config(data_config, config_dir), None


def create_mapper_from_config(mapper_config):
    """根据配置创建映射器"""
    params = mapper_config.get_mapper_params()
    
    if mapper_config.type == 'rank_based':
        return RankBasedMapper(**params)
    elif mapper_config.type == 'direct':
        return DirMapper(**params)
    elif mapper_config.type == 'quantile':
        return QuantileMapper(**params)
    elif mapper_config.type == 'zscore':
        return ZScoreMapper(**params)
    else:
        raise ValueError(f"未知的映射器类型: {mapper_config.type}")


def create_storage_config(config) -> StorageConfig:
    """
    根据配置创建存储配置
    
    支持两种存储模式：
    - 'file': 文件存储模式（默认），保存图片和指标结果到文件系统
    - 'database': 数据库存储模式，只保存指标结果到 SQLite
    """
    storage_config_data = config.storage
    
    # 检查配置中是否指定了存储模式
    storage_mode = getattr(storage_config_data, 'storage_mode', 'file')
    
    if storage_mode == 'database':
        # 数据库模式
        db_path = getattr(storage_config_data, 'db_path', r"E:\code_project\factor_eval_result\factor_eval.db")
        return StorageConfig(
            storage_mode='database',
            db_path=db_path,
        )
    else:
        # 文件模式（默认）
        return StorageConfig(
            storage_mode='file',
            base_path=storage_config_data.base_path,
            save_metrics=storage_config_data.save_metrics,
            save_config=storage_config_data.save_config,
            save_report=storage_config_data.save_report,
            save_plots=storage_config_data.save_plots,
            save_comparison=storage_config_data.save_comparison,
        )


def run_evaluation_from_config(config_path: str):
    """
    从配置文件运行因子评估
    
    Args:
        config_path: 配置文件路径
    """
    print("="*70)
    print("ETF因子评估系统 - 配置化运行")
    print("="*70)
    
    # ========== 1. 加载配置 ==========
    print(f"\n[1/5] 加载配置文件: {config_path}")
    config = load_config(config_path)
    print(f"  - 配置名称: {config.name}")
    print(f"  - 配置版本: {config.version}")
    if config.description:
        print(f"  - 配置描述: {config.description}")
    
    # ========== 2. 验证配置 ==========
    print(f"\n[2/5] 验证配置")
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate(config)
    
    if not is_valid:
        print("  [ERROR] 配置验证失败:")
        for error in errors:
            print(f"    - {error}")
        return
    
    print("  [OK] 配置验证通过")
    if warnings:
        print("  [WARN] 警告:")
        for warning in warnings:
            print(f"    - {warning}")
    
    # ========== 3. 加载数据 ==========
    print(f"\n[3/5] 加载数据")
    config_dir = os.path.dirname(os.path.abspath(config_path))
    ohlcv, trade_context = load_data_from_config(config.data, config_dir)
    data_source = getattr(config.data, 'data_source', 'csv')
    if data_source == 'duckdb':
        print(f"  - 数据源: DuckDB (A股)")
        # 将 evaluation 配置中的 suspended_value_mode 同步到 TradeContext
        if trade_context is not None:
            trade_context.suspended_value_mode = getattr(
                config.evaluation, 'suspended_value_mode', 'freeze'
            )
    else:
        print(f"  - 数据源: CSV (ETF)")
    
    # ========== 4. 运行因子评估 ==========
    print(f"\n[4/5] 运行因子评估")
    print(f"  - 因子数量: {len(config.factors)}")
    
    # 创建存储器
    storage_config = create_storage_config(config)
    storage = ResultStorage(storage_config)
    
    # 判断存储模式
    is_database_mode = storage_config.is_database_mode()
    if is_database_mode:
        print(f"  - 存储模式: 数据库模式")
        print(f"  - 数据库路径: {storage_config.db_path}")
    else:
        print(f"  - 存储模式: 文件模式")
        print(f"  - 基础路径: {storage_config.base_path}")
    
    all_results = []
    
    for factor_config in config.factors:
        print(f"\n  --- 评估因子: {factor_config.name} ---")
        
        # 4.1 计算因子
        print(f"    计算因子: {factor_config.type}")
        factor_calc = create_factor(factor_config.type, **factor_config.params)
        factor_data = factor_calc.calculate(ohlcv)
        # 确保 factor_type 被设置（从配置中的 type 字段）
        if factor_data.factor_type == factor_data.name or factor_data.factor_type == 'unknown':
            factor_data.factor_type = factor_config.type
        
        # 4.2 仓位映射
        print(f"    仓位映射: {config.mapper.type}")
        mapper = create_mapper_from_config(config.mapper)
        position_data = mapper.map_to_position(factor_data)
        
        # 4.3 创建评估器
        eval_cfg = config.evaluation
        evaluator = FactorEvaluator(
            factor_data=factor_data,
            ohlcv_data=ohlcv,
            position_data=position_data,
            forward_period=eval_cfg.forward_period,
            periods_per_year=eval_cfg.periods_per_year,
            risk_free_rate=eval_cfg.risk_free_rate,
            commission_rate=eval_cfg.commission_rate,
            slippage_rate=eval_cfg.slippage_rate,
            delay=eval_cfg.delay,
            rebalance_freq=eval_cfg.rebalance_freq,
            # A股扩展参数（旧配置中这些字段取默认值，不影响原有行为）
            execution_price=getattr(eval_cfg, 'execution_price', 'close'),
            trade_context=trade_context,
            buy_commission_rate=getattr(eval_cfg, 'buy_commission_rate', None),
            sell_commission_rate=getattr(eval_cfg, 'sell_commission_rate', None),
            stamp_tax_rate=getattr(eval_cfg, 'stamp_tax_rate', 0.0),
        )
        
        # 4.4 运行评估
        print(f"    运行评估...")
        results = evaluator.run_full_evaluation()
        all_results.append(results)
        
        # 打印关键指标
        ic_metrics = results.get('ic_metrics', {})
        risk_adj_metrics = results.get('risk_adjusted_metrics', {})
        print(f"    Rank IC: {ic_metrics.get('rank_ic_mean', 0):.4f}")
        print(f"    Sharpe:  {risk_adj_metrics.get('sharpe_ratio', 0):.4f}")
        
        # 4.5 保存结果
        print(f"    保存结果...")
        
        if is_database_mode:
            # 数据库模式：只保存指标，不生成图表
            dataset_name = get_dataset_name_from_config(config)
            _ds_params = {
                'config_name': config.name,
                'config_version': config.version,
                'data_source': getattr(config.data, 'data_source', 'csv'),
                'evaluated_at': datetime.now().isoformat(),
            }
            if getattr(config.data, 'data_source', 'csv') == 'csv':
                _ds_params['csv_path'] = getattr(config.data, 'csv_path', None)
            else:
                _ds_params['start_date'] = getattr(config.data, 'start_date', None)
                _ds_params['end_date'] = getattr(config.data, 'end_date', None)
            record_id = storage.save_evaluation_result(
                result=results,
                factor_name=factor_config.name,
                factor_type=factor_config.type,
                params=factor_config.params,
                dataset_name=dataset_name,
                dataset_params=_ds_params,
                mapper_config=config.mapper.to_dict(),
                evaluation_config=config.evaluation.to_dict(),
            )
            print(f"    记录ID: {record_id}")
        else:
            # 文件模式：生成图表和报告
            report = evaluator.generate_report()
            figures = evaluator.plot()
            
            saved_files = storage.save_evaluation_result(
                result=results,
                figures=figures,
                report=report,
                config={
                    'factor': factor_config.to_dict(),
                    'mapper': config.mapper.to_dict(),
                    'evaluation': config.evaluation.to_dict(),
                },
                factor_name=factor_config.name,
                params=factor_config.params
            )
            print(f"    保存文件数: {len(saved_files)}")
    
    # ========== 5. 生成对比报告 ==========
    print(f"\n[5/5] 生成对比报告")
    if len(all_results) > 1:
        comparison_data = []
        for result in all_results:
            row = {
                '因子': result.get('factor_name', 'Unknown'),
                'IC均值': result['ic_metrics'].get('ic_mean', 0),
                'Rank IC': result['ic_metrics'].get('rank_ic_mean', 0),
                'Rank ICIR': result['ic_metrics'].get('rank_ic_ir', 0),
                '年化收益': result['returns_metrics'].get('annualized_return', 0),
                '夏普比率': result['risk_adjusted_metrics'].get('sharpe_ratio', 0),
                '最大回撤': result['risk_metrics'].get('max_drawdown', 0),
                'Calmar': result['risk_adjusted_metrics'].get('calmar_ratio', 0),
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n对比表格:")
        print(comparison_df.to_string(index=False))
        
        if not is_database_mode:
            comparison_path = storage.save_comparison_table(comparison_df)
            print(f"\n对比表已保存: {comparison_path}")
        else:
            print(f"\n对比表格（数据库存储模式不保存对比表文件）")
    
    print("\n" + "="*70)
    print("评估完成!")
    if is_database_mode:
        print(f"结果已保存到数据库: {storage_config.db_path}")
    else:
        print(f"结果保存路径: {config.storage.base_path}")
    print("="*70)
    
    # 如果是数据库模式，显示统计信息
    if is_database_mode:
        print("\n数据库统计:")
        db = DatabaseStorage(storage_config.db_path)
        stats = db.get_statistics()
        print(f"  总记录数: {stats['total_records']}")
        print(f"  不同表达式: {stats['distinct_expressions']}")
        print(f"  不同数据集: {stats['distinct_datasets']}")


def demo_create_and_save_config():
    """演示创建和保存配置（文件存储模式）"""
    print("\n" + "="*70)
    print("演示: 创建和保存配置（文件存储模式）")
    print("="*70)
    
    # 获取项目根目录（从examples目录往上两级到项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(project_root, "etf_rotation_daily.csv")
    
    # 创建配置
    config = Config.create_simple_config(
        csv_path=csv_path,
        factor_name="RSI_10",
        factor_type="RSI",
        factor_params={"period": 10},
        mapper_type="rank_based",
        mapper_params={"top_k": 5, "direction": -1}
    )
    config.name = "RSI演示配置"
    config.description = "使用RSI因子进行演示评估"
    config.evaluation.forward_period = 5
    
    # 添加更多因子
    from config import FactorConfig
    config.add_factor(FactorConfig(
        name="CloseOverMA_20",
        type="CloseOverMA",
        params={"period": 20}
    ))
    
    # 验证配置
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate(config)
    
    print(f"\n配置验证结果:")
    print(f"  - 有效: {is_valid}")
    if errors:
        print(f"  - 错误: {errors}")
    if warnings:
        print(f"  - 警告: {warnings}")
    
    # 保存配置
    import tempfile
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "demo_config.json")
    config.save_json(config_path)
    
    print(f"\n配置已保存到: {config_path}")
    print(f"\n配置内容预览:")
    print(config.to_json(indent=2)[:500] + "...")
    
    return config_path


def demo_create_and_save_config_database():
    """演示创建和保存配置（数据库存储模式）"""
    print("\n" + "="*70)
    print("演示: 创建和保存配置（数据库存储模式）")
    print("="*70)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(project_root, "etf_rotation_daily.csv")
    
    # 创建配置
    config = Config.create_simple_config(
        csv_path=csv_path,
        factor_name="RSI_14",
        factor_type="RSI",
        factor_params={"period": 14},
        mapper_type="rank_based",
        mapper_params={"top_k": 5, "direction": -1}
    )
    config.name = "RSI数据库存储演示"
    config.description = "使用RSI因子进行演示评估，结果保存到数据库"
    config.evaluation.forward_period = 5
    
    # 关键：设置数据库存储模式
    config.storage.storage_mode = 'database'
    config.storage.db_path = r"E:\code_project\factor_eval_result\factor_eval.db"
    
    # 添加数据集名称（用于数据库查询）
    config.dataset_name = "ETF_Rotation_Daily"
    
    # 验证配置
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate(config)
    
    print(f"\n配置验证结果:")
    print(f"  - 有效: {is_valid}")
    print(f"  - 存储模式: {config.storage.storage_mode}")
    print(f"  - 数据库路径: {config.storage.db_path}")
    if errors:
        print(f"  - 错误: {errors}")
    if warnings:
        print(f"  - 警告: {warnings}")
    
    # 保存配置
    import tempfile
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "demo_config_db.json")
    config.save_json(config_path)
    
    print(f"\n配置已保存到: {config_path}")
    print(f"\n配置内容预览:")
    print(config.to_json(indent=2))
    
    return config_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用配置文件运行因子评估')
    parser.add_argument(
        '--config', 
        type=str,
        help='配置文件路径'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行演示模式（文件存储，创建临时配置并运行）'
    )
    parser.add_argument(
        '--demo-db',
        action='store_true',
        help='运行演示模式（数据库存储，创建临时配置并运行）'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # 文件存储演示模式
        config_path = demo_create_and_save_config()
        try:
            run_evaluation_from_config(config_path)
        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(os.path.dirname(config_path))
    elif args.demo_db:
        # 数据库存储演示模式
        config_path = demo_create_and_save_config_database()
        try:
            run_evaluation_from_config(config_path)
        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(os.path.dirname(config_path))
    elif args.config:
        # 从指定配置运行
        run_evaluation_from_config(args.config)
    else:
        # 使用示例配置
        example_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "examples", "single_factor.json"
        )
        if os.path.exists(example_config):
            print(f"使用示例配置: {example_config}")
            run_evaluation_from_config(example_config)
        else:
            print("错误: 未找到配置文件")
            print(f"请指定 --config 参数或运行 --demo / --demo-db 模式")
            print(f"\n示例:")
            print(f"  python run_from_config.py --config config/examples/single_factor.json")
            print(f"  python run_from_config.py --demo")
            print(f"  python run_from_config.py --demo-db  # 数据库存储模式")


if __name__ == "__main__":
    main()
